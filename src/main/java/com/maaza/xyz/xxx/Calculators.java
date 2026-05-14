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
			final float fact = 2.0f / size;
			float loss = 0.0f;
			for (int i = 0; i < size; i++) {
				final float dif = pred[i] - real[i];
				loss += dif * dif;
				grad[i] = dif * fact;
			}
			return loss / size;
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
				float sig = 0.0f;
				if (p >= 0.0f) {
					sig = 1.0f / (1.0f + exp);
				} else {
					sig = exp / (1.0f + exp);
				}
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
				float max = pred[off];
				for (int i = 1; i < cols; i++) {
					final int k = off + i;
					max = Math.max(max, pred[k]);
				}
				float sum = 0.0f;
				for (int i = 0; i < cols; i++) {
					final int k = off + i;
					final float exp = (float) Math.exp(pred[k] - max);
					grad[k] = exp;
					sum += exp;
				}
				final float log = (float) Math.log(sum);
				final float mul = 1.0f / sum;
				for (int i = 0; i < cols; i++) {
					final int k = off + i;
					final float smx = grad[k] * mul;
					if (real[k] > 0.0f) {
						loss -= real[k] * ((pred[k] - max) - log);
					}
					grad[k] = (smx - real[k]) * fact;
				}
			}
			return loss * fact;
		};
	}
}
