// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Clip shape types for layer clipping.

/// A shape used to clip a layer's content and descendants.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClipShape {
    /// An axis-aligned rectangle.
    Rect(kurbo::Rect),
    /// A rectangle with rounded corners.
    RoundedRect(kurbo::RoundedRect),
}

impl ClipShape {
    /// Returns whether `point` lies inside this clip shape.
    #[must_use]
    pub fn contains(&self, point: kurbo::Point) -> bool {
        match self {
            Self::Rect(r) => r.contains(point),
            Self::RoundedRect(rr) => {
                use kurbo::Shape;
                rr.contains(point)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kurbo::{Point, Rect, RoundedRect};

    #[test]
    fn rect_contains_inside() {
        let clip = ClipShape::Rect(Rect::new(10.0, 10.0, 100.0, 100.0));
        assert!(clip.contains(Point::new(50.0, 50.0)));
    }

    #[test]
    fn rect_rejects_outside() {
        let clip = ClipShape::Rect(Rect::new(10.0, 10.0, 100.0, 100.0));
        assert!(!clip.contains(Point::new(5.0, 50.0)));
        assert!(!clip.contains(Point::new(50.0, 105.0)));
    }

    #[test]
    fn rounded_rect_contains_center() {
        let clip = ClipShape::RoundedRect(RoundedRect::from_rect(
            Rect::new(0.0, 0.0, 100.0, 100.0),
            20.0,
        ));
        assert!(clip.contains(Point::new(50.0, 50.0)));
    }

    #[test]
    fn rounded_rect_rejects_corner() {
        // Point in the bounding rect but outside the rounded corner arc.
        let clip = ClipShape::RoundedRect(RoundedRect::from_rect(
            Rect::new(0.0, 0.0, 100.0, 100.0),
            20.0,
        ));
        assert!(!clip.contains(Point::new(2.0, 2.0)));
    }

    #[test]
    fn rounded_rect_rejects_outside() {
        let clip = ClipShape::RoundedRect(RoundedRect::from_rect(
            Rect::new(0.0, 0.0, 100.0, 100.0),
            10.0,
        ));
        assert!(!clip.contains(Point::new(-5.0, 50.0)));
    }
}
