// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Internal bounded queue utilities.

use std::collections::VecDeque;

/// Bounded FIFO queue with a `drop_oldest` overflow policy.
///
/// Once full, new pushes remove the oldest item before inserting the newest.
#[derive(Debug, Clone)]
pub(crate) struct BoundedQueue<T> {
    items: VecDeque<T>,
    capacity: usize,
    dropped_count: u64,
}

impl<T> BoundedQueue<T> {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            items: VecDeque::with_capacity(capacity),
            capacity,
            dropped_count: 0,
        }
    }

    pub(crate) fn push(&mut self, item: T) {
        if self.items.len() == self.capacity {
            let _ = self.items.pop_front();
            self.dropped_count += 1;
        }
        self.items.push_back(item);
    }

    pub(crate) fn pop(&mut self) -> Option<T> {
        self.items.pop_front()
    }

    pub(crate) fn len(&self) -> usize {
        self.items.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub(crate) fn dropped_count(&self) -> u64 {
        self.dropped_count
    }
}

#[cfg(test)]
mod tests {
    use super::BoundedQueue;

    #[test]
    fn zero_capacity_is_promoted_to_one() {
        let mut queue = BoundedQueue::with_capacity(0);
        queue.push(10_u32);
        queue.push(11_u32);

        assert_eq!(queue.len(), 1);
        assert_eq!(queue.pop(), Some(11_u32));
        assert_eq!(queue.dropped_count(), 1);
    }

    #[test]
    fn push_over_capacity_drops_oldest() {
        let mut queue = BoundedQueue::with_capacity(2);
        queue.push(1_u32);
        queue.push(2_u32);
        queue.push(3_u32);

        assert_eq!(queue.pop(), Some(2_u32));
        assert_eq!(queue.pop(), Some(3_u32));
        assert_eq!(queue.pop(), None);
        assert_eq!(queue.dropped_count(), 1);
    }

    #[test]
    fn empty_queue_reports_is_empty() {
        let mut queue = BoundedQueue::with_capacity(2);
        assert!(queue.is_empty());

        queue.push(1_u32);
        assert!(!queue.is_empty());

        let _ = queue.pop();
        assert!(queue.is_empty());
    }
}
