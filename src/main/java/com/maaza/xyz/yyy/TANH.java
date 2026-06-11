package com.maaza.xyz.yyy;

import com.maaza.xyz.xxx.Array;
import com.maaza.xyz.xxx.Param;

public final class TANH implements Layer {

	private static final Param[] PARAMS = {};

	private Array cache;

	@Override
	public final void forward(final Array input, final Array output) {
		this.cache = output;
		final float[] x = input.data;
		final float[] y = output.data;
		final int size = x.length;
		for (int i = 0; i < size; i++) {
			y[i] = (float) Math.tanh(x[i]);
		}
	}

	@Override
	public final void backward(final Array output, final Array input) {
		final float[] y = this.cache.data;
		final float[] dy = output.data;
		final float[] dx = input.data;
		final int size = dy.length;
		for (int i = 0; i < size; i++) {
			final float yi = y[i];
			dx[i] += dy[i] * (1.0f - yi * yi);
		}
		this.cache = null;
	}

	@Override
	public final int[] shape(final int... size) {
		return size.clone();
	}

	@Override
	public final Param[] params() {
		return PARAMS;
	}
}
