package com.maaza.xyz.yyy;

import com.maaza.xyz.xxx.Array;
import com.maaza.xyz.xxx.Param;

public final class SIGM implements Layer {

	private static final Param[] PARAMS = {};

	private Array cache;

	@Override
	public final void forward(final Array input, final Array output) {
		this.cache = output;
		final float[] x = input.data;
		final float[] y = output.data;
		for (int i = 0; i < x.length; i++) {
			y[i] = 1.0f / (1.0f + (float) Math.exp(-x[i]));
		}
	}

	@Override
	public final void backward(final Array output, final Array input) {
		final float[] y = this.cache.data;
		final float[] dy = output.data;
		final float[] dx = input.data;
		for (int i = 0; i < y.length; i++) {
			final float yi = y[i];
			dx[i] = dy[i] * yi * (1.0f - yi);
		}
	}

	@Override
	public final Param[] params() {
		return PARAMS;
	}
}
