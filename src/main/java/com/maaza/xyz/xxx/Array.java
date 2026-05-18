package com.maaza.xyz.xxx;

public final class Array {

	public final int[] size;
	public final float[] data;

	public Array(final int... size) {
		this.size = size.clone();
		int span = 1;
		for (final int item : size) {
			span *= item;
		}
		this.data = new float[span];
	}
}
