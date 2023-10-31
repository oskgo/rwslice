use core::cmp::min;
use core::marker::PhantomData;
use core::ops::Deref;
use core::ops::DerefMut;
use core::slice;
use core::sync::atomic::AtomicIsize;
use core::sync::atomic::Ordering;

pub struct SRWBuffer<'a, T> {
    boundary: AtomicIsize,
    buffer: &'a mut [T]
}

impl<'a, T> SRWBuffer<'a, T> {
    pub fn new(b: &'a mut [T]) -> Self {
        SRWBuffer {
            boundary: AtomicIsize::new(0),
            buffer: b,
        }
    }

    pub fn split(&mut self) -> (MutSRWBuffer<'_, T>, RefSRWBuffer<'_, T>) {
        let boundary = &self.boundary;
        let len = self.buffer.len();
        let buffer = self.buffer.as_mut_ptr();
        (
            MutSRWBuffer {
                boundary,
                len,
                buffer,
                _ph: PhantomData,
            },
            RefSRWBuffer {
                boundary,
                buffer,
                _ph: PhantomData,
            },
        )
    }
}

// Invariant: Only has mutable access to portion of buffer after the boundary
// Only one MutSplitBuffer with a given buffer exists
pub struct MutSRWBuffer<'a, T> {
    boundary: &'a AtomicIsize,
    len: usize,
    buffer: *mut T,
    _ph: PhantomData<&'a mut [T]>,
}

// SAFETY: The existence of a reference means there is no writer
unsafe impl<'a, T: Sync> Sync for MutSRWBuffer<'a, T> {}
// SAFETY: Writes are hidden by the boundary, and reads cannot see the area before the boundary before the writes have been finished
unsafe impl<'a, T: Send> Send for MutSRWBuffer<'a, T> {}

impl<'a, T> MutSRWBuffer<'a, T> {
    pub fn shrink(&mut self, n: usize) {
        let boundary = self.boundary();
        let new_boundary = min(self.len as isize, boundary + (n as isize));
        self.boundary.store(new_boundary, Ordering::Release); // This Release is paired with the Acquire of the shared buffer
    }

    pub fn boundary(&self) -> isize {
        self.boundary.load(Ordering::Relaxed) // All the stores come from the thread with ownership of this object, so Relaxed is sufficient here
    }
}

// TODO: This can technically give access to the whole buffer. Should it?
// It would be inconsistent, and the indexing would be different.
impl<'a, T> Deref for MutSRWBuffer<'a, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        let boundary = self.boundary();
        // SAFETY: We only give access to the portion of the buffer after the boundary
        unsafe {
            slice::from_raw_parts(self.buffer.offset(boundary), self.len - (boundary as usize))
        }
    }
}

impl<'a, T> DerefMut for MutSRWBuffer<'a, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        let boundary = self.boundary();
        // SAFETY: We only give access to the portion of the buffer after the boundary
        unsafe {
            slice::from_raw_parts_mut(self.buffer.offset(boundary), self.len - (boundary as usize))
        }
    }
}

// Invariant: We only get shared access to the portion of the buffer before the boundary
#[derive(Clone, Copy)]
pub struct RefSRWBuffer<'a, T> {
    boundary: &'a AtomicIsize,
    buffer: *const T,
    _ph: PhantomData<&'a [T]>,
}

// SAFETY: We only get to see parts of the buffer where no more writes are done
unsafe impl<'a, T: Sync> Sync for RefSRWBuffer<'a, T> {}
// SAFETY: We only get to see parts of the buffer where no more writes are done and we only get immutable access
unsafe impl<'a, T: Sync> Send for RefSRWBuffer<'a, T> {}

impl<'a, T> Deref for RefSRWBuffer<'a, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        let boundary = self.boundary.load(Ordering::Acquire); // This has to use Acquire to not see mutations of the buffer after the boundary is released
        // SAFETY: We only give access to the portion before the boundary
        unsafe { slice::from_raw_parts(self.buffer, boundary as usize) }
    }
}

#[test]
fn test_split_buffer() {
    use std::thread::scope;
    use std::thread::sleep;
    use core::time::Duration;
    let b = &mut [0, 0, 0, 0];
    let mut sb = SRWBuffer::new(&mut *b);
    scope(|s| {
        let (mut m, r) = sb.split();
        s.spawn(move || {
            m[1] = 1;
            m.shrink(2);
            sleep(Duration::from_millis(100));
            m.shrink(2);
            assert!(m.is_empty());
        });
        s.spawn(move || {
            sleep(Duration::from_millis(50));
            assert_eq!([0, 1], *r);
            sleep(Duration::from_millis(100));
            assert_eq!([0, 1, 0, 0], *r);
        });
    });
}


