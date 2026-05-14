package com.maaza.xyz.xxx;

public final class Optimizers {

	private Optimizers() {
	}

	private static final class MntmState {

		private final float[] v;

		private MntmState(final int size) {
			this.v = new float[size];
		}
	}

	private static final class AdamState {

		private final float[] m;
		private final float[] v;

		private AdamState(final int size) {
			this.m = new float[size];
			this.v = new float[size];
		}
	}

	public static final Optimizer mntm(final float rate, final float beta) {
		final float bm = 1.0f - beta;
		return (x) -> {
			for (final Param item : x) {
				final float[] data = item.data.data;
				final float[] grad = item.grad.data;
				final int size = data.length;
				if (item.state == null) {
					item.state = new MntmState(size);
				}
				final MntmState state = (MntmState) item.state;
				final float[] v = state.v;
				for (int i = 0; i < size; i++) {
					final float vi = v[i] = beta * v[i] + bm * grad[i];
					data[i] -= rate * vi;
					grad[i] = 0.0f;
				}
			}
		};
	}

	public static final Optimizer adam(final float rate, final float beta1, final float beta2) {
		final float eps = 1.0e-8f;
		final float b1m = 1.0f - beta1;
		final float b2m = 1.0f - beta2;
		final float[] bt = { 1.0f, 1.0f };
		return (x) -> {
			final float b1t = bt[0] *= beta1;
			final float b2t = bt[1] *= beta2;
			final float alpha = rate * (float) Math.sqrt(1.0f - b2t) / (1.0f - b1t);
			for (final Param item : x) {
				final float[] data = item.data.data;
				final float[] grad = item.grad.data;
				final int size = data.length;
				if (item.state == null) {
					item.state = new AdamState(size);
				}
				final AdamState state = (AdamState) item.state;
				final float[] m = state.m;
				final float[] v = state.v;
				for (int i = 0; i < size; i++) {
					final float gi = grad[i];
					final float mi = m[i] = beta1 * m[i] + b1m * gi;
					final float vi = v[i] = beta2 * v[i] + b2m * gi * gi;
					data[i] -= alpha * mi / ((float) Math.sqrt(vi) + eps);
					grad[i] = 0.0f;
				}
			}
		};
	}
}
