// Copyright 2026 the Subduction Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Minimal column-major 4×4 transform.
//!
//! This type covers the subset of 3-D affine transforms that `subduction_core`
//! actually needs (identity, multiply, column access, field mutation) without
//! pulling in a full linear-algebra crate.

use core::ops::Mul;

/// A column-major 4×4 affine transform stored as `[[f64; 4]; 4]`.
///
/// Each inner array is one *column* of the matrix, matching the memory layout
/// used by GPU APIs and Core Animation's `CATransform3D`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform3d {
    /// Four columns, each a 4-element array `[x, y, z, w]`.
    pub cols: [[f64; 4]; 4],
}

impl Transform3d {
    /// The 4×4 identity matrix.
    pub const IDENTITY: Self = Self {
        cols: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };

    /// Creates a transform from four column arrays.
    #[inline]
    #[must_use]
    pub const fn from_cols(col0: [f64; 4], col1: [f64; 4], col2: [f64; 4], col3: [f64; 4]) -> Self {
        Self {
            cols: [col0, col1, col2, col3],
        }
    }

    /// Creates a transform from a column-major 2-D array.
    #[inline]
    #[must_use]
    pub const fn from_cols_array_2d(cols: [[f64; 4]; 4]) -> Self {
        Self { cols }
    }

    /// Returns the columns as a 2-D array.
    #[inline]
    #[must_use]
    pub const fn to_cols_array_2d(self) -> [[f64; 4]; 4] {
        self.cols
    }

    /// Returns column `i` (0-based).
    ///
    /// # Panics
    ///
    /// Panics if `i >= 4`.
    #[inline]
    #[must_use]
    pub const fn col(self, i: usize) -> [f64; 4] {
        self.cols[i]
    }

    /// Creates a pure translation transform.
    #[inline]
    #[must_use]
    pub const fn from_translation(x: f64, y: f64, z: f64) -> Self {
        Self {
            cols: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [x, y, z, 1.0],
            ],
        }
    }

    /// Creates a non-uniform scale transform.
    #[inline]
    #[must_use]
    pub const fn from_scale(sx: f64, sy: f64, sz: f64) -> Self {
        Self {
            cols: [
                [sx, 0.0, 0.0, 0.0],
                [0.0, sy, 0.0, 0.0],
                [0.0, 0.0, sz, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Creates a rotation around the Z axis (radians).
    #[inline]
    #[must_use]
    pub fn from_rotation_z(radians: f64) -> Self {
        #[cfg(feature = "std")]
        let (s, c) = radians.sin_cos();
        #[cfg(not(feature = "std"))]
        let (s, c) = (
            <f64 as kurbo::common::FloatFuncs>::sin(radians),
            <f64 as kurbo::common::FloatFuncs>::cos(radians),
        );
        Self {
            cols: [
                [c, s, 0.0, 0.0],
                [-s, c, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Is this transform [finite]?
    ///
    /// [finite]: f64::is_finite
    #[inline]
    #[must_use]
    pub const fn is_finite(&self) -> bool {
        let c = &self.cols;
        c[0][0].is_finite()
            && c[0][1].is_finite()
            && c[0][2].is_finite()
            && c[0][3].is_finite()
            && c[1][0].is_finite()
            && c[1][1].is_finite()
            && c[1][2].is_finite()
            && c[1][3].is_finite()
            && c[2][0].is_finite()
            && c[2][1].is_finite()
            && c[2][2].is_finite()
            && c[2][3].is_finite()
            && c[3][0].is_finite()
            && c[3][1].is_finite()
            && c[3][2].is_finite()
            && c[3][3].is_finite()
    }

    /// Computes the inverse of this affine 4×4 matrix.
    ///
    /// Inverts the upper-left 3×3 via cofactors, then computes the inverse
    /// translation as −M⁻¹ · t. Returns `None` if the 3×3 determinant is
    /// near zero (absolute value below `1e-12`).
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let c = &self.cols;

        // Extract 3×3 sub-matrix: m[row][col] stored as c[col][row].
        let m00 = c[0][0];
        let m01 = c[1][0];
        let m02 = c[2][0];
        let m10 = c[0][1];
        let m11 = c[1][1];
        let m12 = c[2][1];
        let m20 = c[0][2];
        let m21 = c[1][2];
        let m22 = c[2][2];

        // Cofactors of the 3×3.
        let cf00 = m11 * m22 - m12 * m21;
        let cf01 = m12 * m20 - m10 * m22;
        let cf02 = m10 * m21 - m11 * m20;
        let cf10 = m02 * m21 - m01 * m22;
        let cf11 = m00 * m22 - m02 * m20;
        let cf12 = m01 * m20 - m00 * m21;
        let cf20 = m01 * m12 - m02 * m11;
        let cf21 = m02 * m10 - m00 * m12;
        let cf22 = m00 * m11 - m01 * m10;

        let det = m00 * cf00 + m01 * cf01 + m02 * cf02;
        if det.abs() < 1e-12 {
            return None;
        }

        let inv_det = 1.0 / det;

        // Adjugate (transpose of cofactor) divided by determinant.
        // Column-major: inv_cols[col][row] = adjugate[row][col] / det
        //                                  = cofactor[col][row] / det
        let i00 = cf00 * inv_det;
        let i10 = cf01 * inv_det;
        let i20 = cf02 * inv_det;
        let i01 = cf10 * inv_det;
        let i11 = cf11 * inv_det;
        let i21 = cf12 * inv_det;
        let i02 = cf20 * inv_det;
        let i12 = cf21 * inv_det;
        let i22 = cf22 * inv_det;

        // Inverse translation: −M⁻¹ · t
        let tx = c[3][0];
        let ty = c[3][1];
        let tz = c[3][2];
        let itx = -(i00 * tx + i01 * ty + i02 * tz);
        let ity = -(i10 * tx + i11 * ty + i12 * tz);
        let itz = -(i20 * tx + i21 * ty + i22 * tz);

        Some(Self::from_cols(
            [i00, i10, i20, 0.0],
            [i01, i11, i21, 0.0],
            [i02, i12, i22, 0.0],
            [itx, ity, itz, 1.0],
        ))
    }

    /// Transforms a 2-D point through this matrix.
    ///
    /// The input is treated as `[x, y, 0, 1]` and the output is projected
    /// by dividing by *w*. Returns `None` if the resulting *w* component is
    /// near zero (absolute value below `1e-12`).
    #[must_use]
    pub fn transform_point(&self, point: kurbo::Point) -> Option<kurbo::Point> {
        let [x, y, _, w] = *self * [point.x, point.y, 0.0, 1.0];
        if w.abs() < 1e-12 {
            return None;
        }
        Some(kurbo::Point::new(x / w, y / w))
    }

    /// Is this transform [NaN]?
    ///
    /// [NaN]: f64::is_nan
    #[inline]
    #[must_use]
    pub const fn is_nan(&self) -> bool {
        let c = &self.cols;
        c[0][0].is_nan()
            || c[0][1].is_nan()
            || c[0][2].is_nan()
            || c[0][3].is_nan()
            || c[1][0].is_nan()
            || c[1][1].is_nan()
            || c[1][2].is_nan()
            || c[1][3].is_nan()
            || c[2][0].is_nan()
            || c[2][1].is_nan()
            || c[2][2].is_nan()
            || c[2][3].is_nan()
            || c[3][0].is_nan()
            || c[3][1].is_nan()
            || c[3][2].is_nan()
            || c[3][3].is_nan()
    }
}

impl Default for Transform3d {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl From<kurbo::Affine> for Transform3d {
    #[inline]
    fn from(affine: kurbo::Affine) -> Self {
        let [a, b, c, d, e, f] = affine.as_coeffs();
        Self::from_cols(
            [a, b, 0.0, 0.0],
            [c, d, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [e, f, 0.0, 1.0],
        )
    }
}

impl From<kurbo::TranslateScale> for Transform3d {
    #[inline]
    fn from(ts: kurbo::TranslateScale) -> Self {
        Self::from(kurbo::Affine::from(ts))
    }
}

impl Mul for Transform3d {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a = &self.cols;
        let b = &rhs.cols;
        let mut out = [[0.0_f64; 4]; 4];
        let mut j = 0;
        while j < 4 {
            let mut i = 0;
            while i < 4 {
                out[j][i] =
                    a[0][i] * b[j][0] + a[1][i] * b[j][1] + a[2][i] * b[j][2] + a[3][i] * b[j][3];
                i += 1;
            }
            j += 1;
        }
        Self { cols: out }
    }
}

impl Mul<[f64; 4]> for Transform3d {
    type Output = [f64; 4];

    #[inline]
    fn mul(self, rhs: [f64; 4]) -> [f64; 4] {
        let a = &self.cols;
        [
            a[0][0] * rhs[0] + a[1][0] * rhs[1] + a[2][0] * rhs[2] + a[3][0] * rhs[3],
            a[0][1] * rhs[0] + a[1][1] * rhs[1] + a[2][1] * rhs[2] + a[3][1] * rhs[3],
            a[0][2] * rhs[0] + a[1][2] * rhs[1] + a[2][2] * rhs[2] + a[3][2] * rhs[3],
            a[0][3] * rhs[0] + a[1][3] * rhs[1] + a[2][3] * rhs[2] + a[3][3] * rhs[3],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_identity() {
        assert_eq!(Transform3d::default(), Transform3d::IDENTITY);
    }

    #[test]
    fn identity_multiply() {
        let t = Transform3d::from_translation(1.0, 2.0, 3.0);
        assert_eq!(Transform3d::IDENTITY * t, t);
        assert_eq!(t * Transform3d::IDENTITY, t);
    }

    #[test]
    fn translation_composition() {
        let a = Transform3d::from_translation(1.0, 0.0, 0.0);
        let b = Transform3d::from_translation(0.0, 2.0, 0.0);
        let c = a * b;
        // Combined translation should be (1, 2, 0).
        let col3 = c.col(3);
        assert_eq!(col3, [1.0, 2.0, 0.0, 1.0]);
    }

    #[test]
    fn scale() {
        let s = Transform3d::from_scale(2.0, 3.0, 4.0);
        assert_eq!(s.col(0)[0], 2.0);
        assert_eq!(s.col(1)[1], 3.0);
        assert_eq!(s.col(2)[2], 4.0);
        assert_eq!(s.col(3), [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn round_trip_cols_array_2d() {
        let t = Transform3d::from_translation(5.0, 6.0, 7.0);
        let arr = t.to_cols_array_2d();
        assert_eq!(Transform3d::from_cols_array_2d(arr), t);
    }

    #[test]
    fn scale_then_translate() {
        let s = Transform3d::from_scale(2.0, 2.0, 2.0);
        let t = Transform3d::from_translation(3.0, 4.0, 0.0);
        // Scale first, then translate: T * S
        let combined = t * s;
        // Column 0 should be scaled.
        assert_eq!(combined.col(0), [2.0, 0.0, 0.0, 0.0]);
        // Translation column should be unchanged (translation applied after).
        assert_eq!(combined.col(3), [3.0, 4.0, 0.0, 1.0]);
    }

    #[test]
    fn rotation_z_ninety_degrees() {
        let r = Transform3d::from_rotation_z(core::f64::consts::FRAC_PI_2);
        // cos=0, sin=1 for +90deg.
        let eps = 1e-6;
        assert!((r.col(0)[0] - 0.0).abs() < eps);
        assert!((r.col(0)[1] - 1.0).abs() < eps);
        assert!((r.col(1)[0] + 1.0).abs() < eps);
        assert!((r.col(1)[1] - 0.0).abs() < eps);
    }

    #[test]
    fn identity_is_finite() {
        assert!(Transform3d::IDENTITY.is_finite());
        assert!(!Transform3d::IDENTITY.is_nan());
    }

    #[test]
    fn nan_detected() {
        let mut t = Transform3d::IDENTITY;
        t.cols[2][1] = f64::NAN;
        assert!(!t.is_finite());
        assert!(t.is_nan());
    }

    #[test]
    fn infinity_detected() {
        let mut t = Transform3d::IDENTITY;
        t.cols[0][3] = f64::INFINITY;
        assert!(!t.is_finite());
        assert!(!t.is_nan());
    }

    #[test]
    fn from_affine_embeds_xy_plane_transform() {
        let affine = kurbo::Affine::new([2.0, 3.0, 5.0, 7.0, 11.0, 13.0]);
        let transform = Transform3d::from(affine);

        assert_eq!(
            transform,
            Transform3d::from_cols(
                [2.0, 3.0, 0.0, 0.0],
                [5.0, 7.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [11.0, 13.0, 0.0, 1.0],
            )
        );
    }

    #[test]
    fn from_affine_maps_points_like_kurbo_affine() {
        let affine = kurbo::Affine::new([2.0, 3.0, 5.0, 7.0, 11.0, 13.0]);
        let point = kurbo::Point::new(17.0, 19.0);
        let affine_point = affine * point;
        let transform_point = Transform3d::from(affine) * [point.x, point.y, 0.0, 1.0];

        assert_eq!(affine_point.x, transform_point[0]);
        assert_eq!(affine_point.y, transform_point[1]);
        assert_eq!(transform_point[2], 0.0);
        assert_eq!(transform_point[3], 1.0);
    }

    #[test]
    fn from_translate_scale_matches_affine_embedding() {
        let ts = kurbo::TranslateScale::new(kurbo::Vec2::new(5.0, 6.0), 2.0);
        let transform = Transform3d::from(ts);

        assert_eq!(
            transform,
            Transform3d::from_cols(
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [5.0, 6.0, 0.0, 1.0],
            )
        );
    }

    #[test]
    fn from_translate_scale_maps_points_like_kurbo_translate_scale() {
        let ts = kurbo::TranslateScale::new(kurbo::Vec2::new(5.0, 6.0), 2.0);
        let point = kurbo::Point::new(17.0, 19.0);
        let ts_point = ts * point;
        let transform_point = Transform3d::from(ts) * [point.x, point.y, 0.0, 1.0];

        assert_eq!(ts_point.x, transform_point[0]);
        assert_eq!(ts_point.y, transform_point[1]);
        assert_eq!(transform_point[2], 0.0);
        assert_eq!(transform_point[3], 1.0);
    }

    #[test]
    fn inverse_of_identity() {
        let inv = Transform3d::IDENTITY.inverse().unwrap();
        assert_eq!(inv, Transform3d::IDENTITY);
    }

    #[test]
    fn inverse_of_translation() {
        let t = Transform3d::from_translation(5.0, -3.0, 7.0);
        let inv = t.inverse().unwrap();
        assert_eq!(inv, Transform3d::from_translation(-5.0, 3.0, -7.0));
    }

    #[test]
    fn inverse_of_scale() {
        let s = Transform3d::from_scale(2.0, 4.0, 5.0);
        let inv = s.inverse().unwrap();
        let expected = Transform3d::from_scale(0.5, 0.25, 0.2);
        let eps = 1e-10;
        for col in 0..4 {
            for row in 0..4 {
                assert!(
                    (inv.cols[col][row] - expected.cols[col][row]).abs() < eps,
                    "mismatch at [{col}][{row}]"
                );
            }
        }
    }

    #[test]
    fn inverse_of_rotation_round_trips() {
        let r = Transform3d::from_rotation_z(0.7);
        let inv = r.inverse().unwrap();
        let product = r * inv;
        let eps = 1e-10;
        for col in 0..4 {
            for row in 0..4 {
                let expected = if col == row { 1.0 } else { 0.0 };
                assert!(
                    (product.cols[col][row] - expected).abs() < eps,
                    "mismatch at [{col}][{row}]: {} vs {expected}",
                    product.cols[col][row]
                );
            }
        }
    }

    #[test]
    fn inverse_of_zero_scale_returns_none() {
        let s = Transform3d::from_scale(0.0, 1.0, 1.0);
        assert!(s.inverse().is_none());
    }

    #[test]
    fn inverse_of_compound_transform() {
        let s = Transform3d::from_scale(2.0, 3.0, 1.0);
        let t = Transform3d::from_translation(10.0, 20.0, 0.0);
        let combined = t * s; // scale then translate
        let inv = combined.inverse().unwrap();
        let product = combined * inv;
        let eps = 1e-10;
        for col in 0..4 {
            for row in 0..4 {
                let expected = if col == row { 1.0 } else { 0.0 };
                assert!(
                    (product.cols[col][row] - expected).abs() < eps,
                    "mismatch at [{col}][{row}]"
                );
            }
        }
    }

    #[test]
    fn transform_point_identity() {
        let p = kurbo::Point::new(3.0, 7.0);
        let result = Transform3d::IDENTITY.transform_point(p).unwrap();
        assert_eq!(result, p);
    }

    #[test]
    fn transform_point_translation() {
        let t = Transform3d::from_translation(10.0, 20.0, 0.0);
        let result = t.transform_point(kurbo::Point::new(5.0, 3.0)).unwrap();
        assert_eq!(result, kurbo::Point::new(15.0, 23.0));
    }

    #[test]
    fn transform_point_round_trip() {
        let s = Transform3d::from_scale(2.0, 3.0, 1.0);
        let t = Transform3d::from_translation(10.0, 20.0, 0.0);
        let combined = t * s;
        let inv = combined.inverse().unwrap();

        let original = kurbo::Point::new(7.0, 11.0);
        let transformed = combined.transform_point(original).unwrap();
        let recovered = inv.transform_point(transformed).unwrap();

        let eps = 1e-10;
        assert!((recovered.x - original.x).abs() < eps);
        assert!((recovered.y - original.y).abs() < eps);
    }
}
