package com.maaza.xyz.yyy;

import com.maaza.xyz.xxx.Array;
import com.maaza.xyz.xxx.Param;

public final class RELU implements Layer {

	private static final Param[] PARAMS = {};

	private Array cache;

	@Override
	public final void forward(final Array input, final Array output) {
		this.cache = input;
		final float[] x = input.data;
		final float[] y = output.data;
		for (int i = 0; i < x.length; i++) {
			y[i] = Math.max(0.0f, x[i]);
		}
	}

	@Override
	public final void backward(final Array output, final Array input) {
		final float[] x = this.cache.data;
		final float[] dx = input.data;
		final float[] dy = output.data;
		for (int i = 0; i < x.length; i++) {
			dx[i] = x[i] > 0.0f ? dy[i] : 0.0f;
		}
	}

	@Override
	public final Param[] params() {
		return PARAMS;
	}
}
