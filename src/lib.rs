#![cfg_attr(not(test), no_std)]
#![warn(future_incompatible)]
#![warn(rust_2018_idioms)]

pub mod sync;

use core::cell::Cell;
use core::cmp::min;
use core::marker::PhantomData;
use core::ops::Deref;
use core::ops::DerefMut;
use core::slice;

pub struct RWBuffer<'a, T> {
    boundary: Cell<isize>,
    buffer: &'a mut [T]
}

impl<'a, T> RWBuffer<'a, T> {
    pub fn new(b: &'a mut [T]) -> Self {
        RWBuffer {
            boundary: Cell::new(0),
            buffer: b,
        }
    }

    pub fn split(&mut self) -> (MutRWBuffer<'_, T>, RefRWBuffer<'_, T>) {
        let boundary = &self.boundary;
        let len = self.buffer.len();
        let buffer = self.buffer.as_mut_ptr();
        (
            MutRWBuffer {
                boundary,
                len,
                buffer,
                _ph: PhantomData,
            },
            RefRWBuffer {
                boundary,
                buffer,
                _ph: PhantomData,
            },
        )
    }
}

// Invariant: Only has mutable access to portion of buffer after the boundary
// Only one MutSplitBuffer with a given buffer exists
pub struct MutRWBuffer<'a, T> {
    boundary: &'a Cell<isize>,
    len: usize,
    buffer: *mut T,
    _ph: PhantomData<&'a mut [T]>,
}

impl<'a, T> MutRWBuffer<'a, T> {
    pub fn shrink(&mut self, n: usize) {
        let boundary = self.boundary.get();
        let new_boundary = min(self.len as isize, boundary + (n as isize));
        self.boundary.replace(new_boundary);
    }

    pub fn boundary(&self) -> isize {
        self.boundary.get()
    }
}

// TODO: Maybe this should not be implemented even though it's sound
// SAFETY: The existence of a reference means there is no writer
unsafe impl<'a, T: Sync> Sync for MutRWBuffer<'a, T> {}

// TODO: This can technically give access to the whole buffer. Should it?
// It would be inconsistent, and the indexing would be different.
impl<'a, T> Deref for MutRWBuffer<'a, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        let boundary = self.boundary.get();
        // SAFETY: We only give access to the portion of the buffer after the boundary
        unsafe {
            slice::from_raw_parts(self.buffer.offset(boundary), self.len - (boundary as usize))
        }
    }
}

impl<'a, T> DerefMut for MutRWBuffer<'a, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        let boundary = self.boundary.get();
        // SAFETY: We only give access to the portion of the buffer after the boundary
        unsafe {
            slice::from_raw_parts_mut(self.buffer.offset(boundary), self.len - (boundary as usize))
        }
    }
}

// Invariant: We only get shared access to the portion of the buffer before the boundary
#[derive(Clone, Copy)]
pub struct RefRWBuffer<'a, T> {
    boundary: &'a Cell<isize>,
    buffer: *const T,
    _ph: PhantomData<&'a [T]>,
}

impl<'a, T> Deref for RefRWBuffer<'a, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        let boundary = self.boundary.get();
        // SAFETY: We only give access to the portion before the boundary
        unsafe { slice::from_raw_parts(self.buffer, boundary as usize) }
    }
}

#[test]
fn test_split_buffer() {
    let b = &mut [0, 0, 0, 0];
    let mut sb = RWBuffer::new(&mut *b);
    let (mut m, s) = sb.split();
    m[1] = 1;
    m.shrink(2);
    assert_eq!([0, 1], &*s);
    m.shrink(2);
    assert!(m.is_empty());
    assert_eq!([0, 1, 0, 0], &*s);
}
