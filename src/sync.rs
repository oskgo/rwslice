use core::cmp::min;
use core::marker::PhantomData;
use core::ops::Deref;
use core::ops::DerefMut;
use core::slice;
use core::sync::atomic::AtomicIsize;
use core::sync::atomic::Ordering;

// Invariants: The `boundary` of a `SRWBuffer` is strictly increasing over its lifetime from the perspective of every thread.
//             No thread can cause unsynchronized writes to the parts of `buffer` in front of the `boundary` observed earlier by said thread.
//             At most one thread (the mutator) can mutate `boundary` at any point in time, and only this thread can have
//             direct access to the parts of `buffer` behind the most recent `boundary`. The access permits unsynchronized writes.
//             ("Most recent `boundary`"" makes sense here since this thread is the only mutator of `boundary`)
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

    // The thread with ownership of the `MutSRWBuffer` is the mutator for the `SRWBuffer`
    // The `RefSRWBuffer` is used for shared access to the front of the buffer
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

// The thread with ownership of the `MutSRWBuffer` is the mutator for the `SRWBuffer` used to create it
// Invariants: A given buffer only has a single `MutSRWBuffer`, which has unsynchronized write access to the part behind the most recent `boundary`
//             That is the only way to get unsynchronized write access to that part of the buffer as long as a `MutSRWBuffer` exists
pub struct MutSRWBuffer<'a, T> {
    boundary: &'a AtomicIsize,
    len: usize,
    buffer: *mut T,
    _ph: PhantomData<&'a mut [T]>,
}

// TODO: If we return an empty slice in the `Deref` implementation we don't need a `Sync` bound here, 
//       but that would also make `&MutSRWBuffer` nigh useless. 
// SAFETY: Any unsynchronized mutations have to happen before the `&MutSRWBuffer` is sent to another thread,
//         and (provided `T` is `Sync`) a shared reference cannot be used to cause unsynchronized mutations
unsafe impl<'a, T: Sync> Sync for MutSRWBuffer<'a, T> {}

// SAFETY: Any unsynchronized mutations will only happen to `T` that are behind the current `boundary`,
//         which is unaccessible to other threads until after `boundary` has moved past it.
unsafe impl<'a, T: Send> Send for MutSRWBuffer<'a, T> {}

impl<'a, T> MutSRWBuffer<'a, T> {
    pub fn shrink(&mut self, n: usize) {
        let boundary = self.boundary();
        let new_boundary = min(self.len as isize, boundary + (n as isize));
        // This Release is paired with the Acquire of the `RefSRWBuffer`s, which makes sure that the unsynchronized mutations that happen before this store
        // also happen before the read of the updated `boundary` on the `RefMutSRWBuffer`'s thread, making sure that the shared reference
        // only includes the region where no more unsynchronized mutations will happen
        self.boundary.store(new_boundary, Ordering::Release);
    }

    pub fn boundary(&self) -> isize {
        // All the stores come from the thread with ownership of this `MutSRWBuffer`, so Relaxed is sufficient here
        // since events on the same thread are synchronized, and any movement across threads also synchronizes
        self.boundary.load(Ordering::Relaxed)
    }
}

// TODO: This can technically give access to the whole buffer. Should it?
// It would be inconsistent, and the indexing would be different.
impl<'a, T> Deref for MutSRWBuffer<'a, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        let boundary = self.boundary();
        // SAFETY: The only way to unsynchonously mutate the underlying data of the slice is through a mutable reference to `self`,
        //         which is currently being borrowed, so no such unsynchronized mutation is possible
        unsafe {
            slice::from_raw_parts(self.buffer.offset(boundary), self.len - (boundary as usize))
        }
    }
}

impl<'a, T> DerefMut for MutSRWBuffer<'a, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        let boundary = self.boundary();
        // SAFETY: We only give mutable access to the part of the buffer after the boundary, which this thread is allowed to do,
        //         Since only one `MutSRWBuffer` exists for the underlying data and it is borrowed until the end of the lifetime
        //         of the returned mutable lifetime it has the proper lifetime and is unique.
        unsafe {
            slice::from_raw_parts_mut(self.buffer.offset(boundary), self.len - (boundary as usize))
        }
    }
}

// Invariants: Any read from the buffer should happen after an Aquire load from the boundary.
//             The only parts of the buffer that are accessible are those in front of the loaded boundary,
//             and the accesses are only shared.
pub struct RefSRWBuffer<'a, T> {
    boundary: &'a AtomicIsize,
    buffer: *const T,
    _ph: PhantomData<&'a [T]>,
}

impl<'a, T> Clone for RefSRWBuffer<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for RefSRWBuffer<'a, T> {}

// SAFETY: We only get shared access to the parts of the buffer where no more unsynchronized writes are done
unsafe impl<'a, T: Sync> Sync for RefSRWBuffer<'a, T> {}
// SAFETY: We only get shared access to the parts of the buffer where no more unsynchronized writes are done
unsafe impl<'a, T: Sync> Send for RefSRWBuffer<'a, T> {}

impl<'a, T> Deref for RefSRWBuffer<'a, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        let boundary = self.boundary.load(Ordering::Acquire);
        // SAFETY: Because of the Release/Acquire relationship between this load and the store in the `MutSRWBuffer`
        //         all unsynchronized writes to the region before the boundary happen before the boundary is moved, and thus we
        //         only create a shared reference to the parts of the buffer that will no longer be unsynchronously written to.
        //         Since the boundary is strictly increasing over the lifetime of this `RefSRWBuffer` loading or persisting an
        //         old value (from the perspective of the mutator) will at worst mean we get access to fewer values than we could.
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


