//! Matcher states as the GPU sees them: the transition table's state ids and entries.
//!
//! A device row holds a state id plus the true counters `(room, need)` of its open top string
//! (room -1: unbounded). State ids name canonical states: the full matcher state with only the
//! open top string's counters clamped to the longest token (the mask's exact range). An entry,
//! keyed by `(shape, token)` inside a string (shape: the state with the string's counters zeroed)
//! and by `(state, token)` elsewhere, either
//! - stays in the open string (`stay`): `next` is the shape after the token and `a` the
//!   characters it adds; the device subtracts them and looks the state id up by
//!   (shape, clamped room, clamped need), or
//! - sets everything (`!stay`): `next` is the state id and `(a, b)` the absolute counters of the
//!   string it ends inside (-1, 0 when none).
//!
//! Inside a string the counters decide only whether a token is allowed, which the mask settles.
//! A plain token in a string body needs no entry: same shape, `a` = its characters (`d_plain`).

use rustc_hash::FxHashMap;

use crate::grammar::matcher::State;
use crate::grammar::schema::UNBOUNDED;
use crate::grammar::store::{Store, EOS_ONLY};

/// State id of a rejected reply: its mask is end-of-sequence only
pub const DEAD: u32 = 0;
const BIG: u32 = 1 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry {
    pub next: u32,
    pub stay: bool,
    pub a: i32,
    pub b: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Row {
    pub state: u32,
    pub room: i32,
    pub need: i32,
}

pub struct Device {
    states: Vec<(u32, State)>,
    ids: FxHashMap<(u32, State), u32>,
    mask: Vec<u32>,
    /// Per state: the shape of its open string, -1 outside strings
    shape: Vec<i32>,
    shapes: Vec<(u32, State)>,
    shape_ids: FxHashMap<(u32, State), u32>,
}

impl Default for Device {
    fn default() -> Self {
        Self::new()
    }
}

impl Device {
    pub fn new() -> Device {
        Device {
            states: vec![(u32::MAX, State::string_body(0))],
            ids: FxHashMap::default(),
            mask: vec![EOS_ONLY],
            shape: vec![-1],
            shapes: Vec::new(),
            shape_ids: FxHashMap::default(),
        }
    }
}

fn room_dev(room: u32) -> i32 {
    if room == UNBOUNDED {
        -1
    } else {
        room.min(i32::MAX as u32) as i32
    }
}

fn room_host(room: i32) -> u32 {
    if room < 0 {
        UNBOUNDED
    } else {
        room as u32
    }
}

/// The device's clamp of a row's counters (room -1: unbounded, clamps to `l`)
pub fn clamp(room: i32, need: i32, l: u32) -> (u32, u32) {
    let r = if room < 0 { l } else { (room as u32).min(l) };
    (r, (need.max(0) as u32).min(l))
}

impl Store {
    fn canon(&self, s: &State) -> State {
        match s.top_str() {
            Some((n, r)) => s.with_top_str(n.min(self.longest), r.min(self.longest)),
            None => *s,
        }
    }

    fn intern_state(&mut self, grammar: u32, s: State) -> u32 {
        let c = self.canon(&s);
        if let Some(&id) = self.device.ids.get(&(grammar, c)) {
            return id;
        }
        let mask = self.mask_id(grammar, &c);
        let shape = match c.top_str() {
            Some(_) => self.intern_shape(grammar, &c) as i32,
            None => -1,
        };
        let d = &mut self.device;
        let id = d.states.len() as u32;
        d.states.push((grammar, c));
        d.ids.insert((grammar, c), id);
        d.mask.push(mask);
        d.shape.push(shape);
        id
    }

    fn intern_shape(&mut self, grammar: u32, s: &State) -> u32 {
        let sh = s.with_top_str(0, 0);
        let d = &mut self.device;
        if let Some(&id) = d.shape_ids.get(&(grammar, sh)) {
            return id;
        }
        let id = d.shapes.len() as u32;
        d.shapes.push((grammar, sh));
        d.shape_ids.insert((grammar, sh), id);
        id
    }

    fn row_of(&mut self, grammar: u32, s: State) -> Row {
        let (room, need) = s
            .top_str()
            .map_or((-1, 0), |(n, r)| (room_dev(r), n as i32));
        Row { state: self.intern_state(grammar, s), room, need }
    }

    /// The row a reply of `grammar` starts as
    pub fn d_initial(&mut self, grammar: u32) -> Row {
        let s = State::new(&self.grammars[grammar as usize]);
        self.row_of(grammar, s)
    }

    /// `token` from `row`: the table entry for (row.state, token) and the row after it;
    /// end-of-sequence keeps the row, a rejected token goes to `DEAD`
    pub fn d_step(&mut self, row: Row, token: u32) -> (Entry, Row) {
        self.stats.tokens += 1;
        let keep = Entry { next: row.state, stay: false, a: row.room, b: row.need };
        if row.state == DEAD || self.eos.contains(&token) {
            return (keep, row);
        }
        let (g, cs) = self.device.states[row.state as usize];
        let inside = cs.top_str().is_some();
        let mut s = if inside {
            cs.with_top_str(row.need.max(0) as u32, room_host(row.room))
        } else {
            cs
        };
        let gr = &self.grammars[g as usize];
        let left = match self.tokens.get(token as usize) {
            Some(Some(bytes)) => s.step_token(gr, self.max_ws, bytes),
            _ => None,
        };
        let Some(left) = left else {
            self.stats.rejected += 1;
            let dead = Row { state: DEAD, room: -1, need: 0 };
            let e = Entry { next: DEAD, stay: false, a: -1, b: 0 };
            return (e, dead);
        };
        let after = self.row_of(g, s);
        if inside && !left {
            // the characters the token adds, read off a copy with counters far from any bound
            let mut big = cs.with_top_str(BIG, BIG);
            let bytes = self.tokens[token as usize].as_ref().unwrap();
            big.step_token(&self.grammars[g as usize], self.max_ws, bytes);
            let k = BIG - big.top_str().map_or(BIG, |(_, r)| r);
            let shape = self.intern_shape(g, &s);
            let e = Entry { next: shape, stay: true, a: k as i32, b: 0 };
            return (e, after);
        }
        let e = Entry { next: after.state, stay: false, a: after.room, b: after.need };
        (e, after)
    }

    /// The state id of string shape `shape` with the device's clamped counters
    pub fn d_counters(&mut self, shape: u32, room: i32, need: i32) -> u32 {
        let (g, sh) = self.device.shapes[shape as usize];
        let (r, n) = clamp(room, need, self.longest);
        self.intern_state(g, sh.with_top_str(n, r))
    }

    /// The mask id of state `id`
    pub fn d_mask(&self, id: u32) -> u32 {
        self.device.mask[id as usize]
    }

    /// The shape of state `id`'s open string, -1 outside strings (the table key's first half)
    pub fn d_shape(&self, id: u32) -> i32 {
        self.device.shape[id as usize]
    }

    /// State `id` sits in a string body at a character boundary (plain tokens step it alone)
    pub fn d_body(&self, id: u32) -> bool {
        self.device.states[id as usize].1.body_room().is_some() && id != DEAD
    }

    /// Each token's characters if it is plain, else -1
    pub fn d_plain(&self) -> Vec<i32> {
        self.plain
            .iter()
            .map(|c| c.map_or(-1, |c| c as i32))
            .collect()
    }

    pub fn d_states(&self) -> usize {
        self.device.states.len()
    }

    pub fn d_longest(&self) -> u32 {
        self.longest
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// The device's side: the table, hits applied as the kernel does, misses through the store
    struct Emu {
        table: HashMap<(bool, u32, u32), Entry>,
        by_counters: HashMap<(u32, u32, u32), u32>,
        misses: usize,
    }

    impl Emu {
        fn step(&mut self, st: &mut Store, row: Row, t: u32) -> Row {
            let sh = st.d_shape(row.state);
            let pc = st.d_plain()[t as usize];
            if st.d_body(row.state) && pc >= 0 {
                let e = Entry { next: sh as u32, stay: true, a: pc, b: 0 };
                return self.counters(st, row, e);
            }
            let key = if sh >= 0 {
                (true, sh as u32, t)
            } else {
                (false, row.state, t)
            };
            let Some(&e) = self.table.get(&key) else {
                self.misses += 1;
                let (e, after) = st.d_step(row, t);
                self.table.insert(key, e);
                return after;
            };
            if !e.stay {
                return Row { state: e.next, room: e.a, need: e.b };
            }
            self.counters(st, row, e)
        }

        fn counters(&mut self, st: &mut Store, row: Row, e: Entry) -> Row {
            let room = if row.room < 0 {
                row.room
            } else {
                row.room - e.a
            };
            let need = (row.need - e.a).max(0);
            let (r, n) = clamp(room, need, st.d_longest());
            let state = match self.by_counters.get(&(e.next, r, n)) {
                Some(&id) => id,
                None => {
                    self.misses += 1;
                    let id = st.d_counters(e.next, room, need);
                    self.by_counters.insert((e.next, r, n), id);
                    id
                }
            };
            Row { state, room, need }
        }
    }

    #[test]
    fn device_rows_match_the_matcher() {
        let words = [
            "{\"", "s", "\":\"", "ab", "c", "abc", "\"}", "é", "\\n", "\"", "}",
        ];
        let vocab: Vec<Option<Vec<u8>>> =
            words.iter().map(|w| Some(w.as_bytes().to_vec())).collect();
        let n = vocab.len();
        let schema = r#"{"type":"object","properties":{"s":{"type":"string","minLength":2,"maxLength":40}},"required":["s"]}"#;
        let mut st = Store::new(vocab, vec![], n, 0);
        let g = st.compile(schema).unwrap();
        let mut emu = Emu { table: HashMap::new(), by_counters: HashMap::new(), misses: 0 };
        // replies of different lengths share the table; each step's mask must be the matcher's
        for len in [3usize, 10, 17, 30, 31, 37] {
            let mut body: Vec<u32> = Vec::new();
            let mut chars = 0;
            let mut i = 0;
            while chars < len {
                let (t, c) = [(3, 2), (4, 1), (5, 3), (7, 1), (8, 1)][i % 5];
                if chars + c > len {
                    body.push(4);
                    chars += 1;
                } else {
                    body.push(t);
                    chars += c;
                }
                i += 1;
            }
            let reply: Vec<u32> = [0, 1, 2].into_iter().chain(body).chain([6]).collect();
            let h = st.matcher(g);
            let mut row = st.d_initial(g);
            for &t in &reply {
                let want = st.step(&[h], &[vec![]])[0];
                assert_eq!(st.d_mask(row.state), want, "len {len}");
                row = emu.step(&mut st, row, t);
                st.step(&[h], &[vec![t]]);
            }
            assert_eq!(st.d_mask(row.state), st.step(&[h], &[vec![]])[0]);
            st.release(h);
        }
        assert_eq!(st.stats.rejected, 0);
        // long bodies reuse the clamped ids: far fewer states than characters written
        assert!(st.d_states() < 60, "{} states", st.d_states());
        assert!(emu.misses < 80, "{} misses", emu.misses);
    }
}
