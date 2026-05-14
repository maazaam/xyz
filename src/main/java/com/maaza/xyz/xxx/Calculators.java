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
			final int size = real.length;
			float max = pred[0];
			for (int i = 1; i < size; i++) {
				max = Math.max(max, pred[i]);
			}
			float sum = 0.0f;
			for (int i = 0; i < size; i++) {
				final float exp = (float) Math.exp(pred[i] - max);
				grad[i] = exp;
				sum += exp;
			}
			final float log = (float) Math.log(sum);
			final float fact = 1.0f / sum;
			float loss = 0.0f;
			for (int i = 0; i < size; i++) {
				final float smx = grad[i] * fact;
				if (real[i] > 0.0f) {
					loss -= real[i] * ((pred[i] - max) - log);
				}
				grad[i] = smx - real[i];
			}
			return loss;
		};
	}
}
