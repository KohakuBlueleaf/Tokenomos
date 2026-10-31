//! pyo3 bindings of `store::Store` as `tokenomos._rs.grammar.Store`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::grammar::{device, store};

#[pyclass(module = "tokenomos._rs.grammar")]
struct Store(store::Store);

#[pymethods]
impl Store {
    /// tokens: each token's bytes (None: never allowed), eos: end-of-sequence ids,
    /// vocab: the logits' width, max_ws: whitespace bytes allowed per gap
    #[new]
    fn new(tokens: Vec<Option<Vec<u8>>>, eos: Vec<u32>, vocab: usize, max_ws: u8) -> Self {
        Store(store::Store::new(tokens, eos, vocab, max_ws))
    }

    fn compile(&mut self, schema: &str) -> PyResult<u32> {
        self.0.compile(schema).map_err(PyValueError::new_err)
    }

    fn matcher(&mut self, grammar: u32) -> u32 {
        self.0.matcher(grammar)
    }

    fn release(&mut self, handle: u32) {
        self.0.release(handle)
    }

    fn step(&mut self, py: Python<'_>, handles: Vec<u32>, tokens: Vec<Vec<u32>>) -> Vec<u32> {
        py.allow_threads(|| self.0.step(&handles, &tokens))
    }

    /// The device row `(state, room, need)` a reply of `grammar` starts as
    fn d_initial(&mut self, grammar: u32) -> (u32, i32, i32) {
        let r = self.0.d_initial(grammar);
        (r.state, r.room, r.need)
    }

    /// Each row `(states[i], rooms[i], needs[i])` after `tokens[i]`: per row the table entry
    /// `(next, stay, a, b)` and the row after it `(state, room, need)`
    #[allow(clippy::type_complexity)]
    fn d_step(
        &mut self,
        py: Python<'_>,
        states: Vec<u32>,
        rooms: Vec<i32>,
        needs: Vec<i32>,
        tokens: Vec<u32>,
    ) -> Vec<((u32, bool, i32, i32), (u32, i32, i32))> {
        py.allow_threads(|| {
            (0..states.len())
                .map(|i| {
                    let row = device::Row { state: states[i], room: rooms[i], need: needs[i] };
                    let (e, r) = self.0.d_step(row, tokens[i]);
                    ((e.next, e.stay, e.a, e.b), (r.state, r.room, r.need))
                })
                .collect()
        })
    }

    /// The state id of each string shape with the device's counters
    fn d_counters(&mut self, shapes: Vec<u32>, rooms: Vec<i32>, needs: Vec<i32>) -> Vec<u32> {
        (0..shapes.len())
            .map(|i| self.0.d_counters(shapes[i], rooms[i], needs[i]))
            .collect()
    }

    /// The mask id of each state id
    fn d_masks(&self, ids: Vec<u32>) -> Vec<u32> {
        ids.iter().map(|&i| self.0.d_mask(i)).collect()
    }

    /// The open-string shape of each state id (-1 outside strings)
    fn d_shapes(&self, ids: Vec<u32>) -> Vec<i32> {
        ids.iter().map(|&i| self.0.d_shape(i)).collect()
    }

    /// Whether each state id sits in a string body at a character boundary
    fn d_bodies(&self, ids: Vec<u32>) -> Vec<bool> {
        ids.iter().map(|&i| self.0.d_body(i)).collect()
    }

    /// Each token's characters if it is plain, else -1
    fn d_plain(&self) -> Vec<i32> {
        self.0.d_plain()
    }

    #[getter]
    fn d_states(&self) -> usize {
        self.0.d_states()
    }

    /// The counters' clamp (the longest token in bytes)
    #[getter]
    fn longest(&self) -> u32 {
        self.0.d_longest()
    }

    /// The words of every mask added since the last call (native-endian uint32)
    fn take_new<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let words = self.0.take_new();
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
        PyBytes::new(py, &bytes)
    }

    fn mask<'py>(&self, py: Python<'py>, id: u32) -> Bound<'py, PyBytes> {
        let bytes: Vec<u8> = self
            .0
            .mask(id)
            .iter()
            .flat_map(|w| w.to_ne_bytes())
            .collect();
        PyBytes::new(py, &bytes)
    }

    #[getter]
    fn words(&self) -> usize {
        self.0.words
    }

    #[getter]
    fn count(&self) -> usize {
        self.0.count()
    }

    /// masks built, seconds building them, step calls, seconds in steps, tokens, rejected
    fn stats(&self) -> (u64, f64, u64, f64, u64, u64) {
        let s = self.0.stats;
        (s.masks, s.mask_s, s.steps, s.step_s, s.tokens, s.rejected)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Store>()?;
    m.add("EOS_ONLY", store::EOS_ONLY)?;
    m.add("DEAD", device::DEAD)?;
    Ok(())
}
