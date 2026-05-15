package com.maaza.xyz.xxx;

public final class Calculators {

	private Calculators() {
	}

	public static final Calculator mse() {
		return (x, y, z) -> {
			final float[] real = x.data;
			final float[] pred = y.data;
			final float[] grad = z.data;
			final int size = real.length;
			final float fact = 1.0f / size;
			float loss = 0.0f;
			for (int i = 0; i < size; i++) {
				final float dif = pred[i] - real[i];
				loss += dif * dif;
				grad[i] = dif * fact;
			}
			return 0.5f * loss * fact;
		};
	}

	public static final Calculator bin() {
		return (x, y, z) -> {
			final float[] real = x.data;
			final float[] pred = y.data;
			final float[] grad = z.data;
			final int size = real.length;
			final float fact = 1.0f / size;
			float loss = 0.0f;
			for (int i = 0; i < size; i++) {
				final float r = real[i];
				final float p = pred[i];
				final float max = Math.max(p, 0.0f);
				final float exp = (float) Math.exp(-Math.abs(p));
				final float log = (float) Math.log1p(exp);
				loss += max - p * r + log;
				final float inv = 1.0f / (1.0f + exp);
				final float sig = p >= 0.0f ? inv : exp * inv;
				grad[i] = (sig - r) * fact;
			}
			return loss * fact;
		};
	}

	public static final Calculator mul() {
		return (x, y, z) -> {
			final float[] real = x.data;
			final float[] pred = y.data;
			final float[] grad = z.data;
			final int[] size = x.size;
			final int cols = size[size.length - 1];
			final int rows = real.length / cols;
			final float fact = 1.0f / rows;
			float loss = 0.0f;
			for (int j = 0; j < rows; j++) {
				final int off = j * cols;
				final int end = off + cols;
				float max = pred[off];
				for (int i = off + 1; i < end; i++) {
					max = Math.max(max, pred[i]);
				}
				float sum = 0.0f;
				for (int i = off; i < end; i++) {
					final float exp = (float) Math.exp(pred[i] - max);
					grad[i] = exp;
					sum += exp;
				}
				final float log = (float) Math.log(sum);
				final float inv = 1.0f / sum;
				for (int i = off; i < end; i++) {
					final float r = real[i];
					final float smx = grad[i] * inv;
					if (r > 0.0f) {
						loss += r * (log - (pred[i] - max));
					}
					grad[i] = (smx - r) * fact;
				}
			}
			return loss * fact;
		};
	}
}
