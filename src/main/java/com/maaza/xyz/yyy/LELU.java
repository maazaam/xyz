package com.maaza.xyz.yyy;

import com.maaza.xyz.xxx.Array;
import com.maaza.xyz.xxx.Param;

public final class LELU implements Layer {

	private static final Param[] PARAMS = new Param[0];

	private static final float ALPHA = 0.01f;

	private Array cache;

	@Override
	public final void forward(final Array input, final Array output) {
		this.cache = input;
		final float[] x = input.data;
		final float[] y = output.data;
		final float alpha = ALPHA;
		for (int i = 0; i < x.length; i++) {
			final float xi = x[i];
			y[i] = Math.max(alpha * xi, xi);
		}
	}

	@Override
	public final void backward(final Array output, final Array input) {
		final float[] x = this.cache.data;
		final float[] dx = input.data;
		final float[] dy = output.data;
		final float alpha = ALPHA;
		for (int i = 0; i < x.length; i++) {
			final float dyi = dy[i];
			dx[i] = x[i] > 0.0f ? dyi : alpha * dyi;
		}
	}

	@Override
	public final Param[] params() {
		return PARAMS;
	}
}
