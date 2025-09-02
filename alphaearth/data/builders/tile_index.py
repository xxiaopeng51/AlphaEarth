import math
from typing import Iterator, List, Tuple, Dict, Optional


def lonlat_to_tile_indices(lon: float, lat: float, zoom: int) -> Tuple[int, int]:
	"""
	Convert WGS84 lon/lat to WebMercator tile x/y at a given zoom.
	"""
	lat = max(min(lat, 85.05112878), -85.05112878)
	n = 1 << zoom
	x = int((lon + 180.0) / 360.0 * n)
	y = int(
		(1.0 - math.log(math.tan(math.radians(lat)) + (1 / math.cos(math.radians(lat)))) / math.pi)
		/ 2.0
		* n
	)
	return x, y


def tile_indices_to_bounds(x: int, y: int, zoom: int) -> Tuple[float, float, float, float]:
	"""
	Return tile bounds in lon/lat: (west, south, east, north).
	"""
	n = 1 << zoom
	lon_w = x / n * 360.0 - 180.0
	lon_e = (x + 1) / n * 360.0 - 180.0
	y_n = y / n
	y_s = (y + 1) / n
	lat_n = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y_n))))
	lat_s = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y_s))))
	return lon_w, lat_s, lon_e, lat_n


def tile_id(z: int, x: int, y: int) -> str:
	return f"z{z}/x{x}/y{y}"


def iterate_tiles(
	zoom: int,
	bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Iterator[Tuple[int, int, int]]:
	"""
	Iterate over WebMercator tiles at zoom within bbox (lon_w, lat_s, lon_e, lat_n).
	If bbox is None, iterate the full world coverage at the zoom.
	"""
	if bbox is None:
		min_lon, min_lat, max_lon, max_lat = -180.0, -85.05112878, 180.0, 85.05112878
	else:
		min_lon, min_lat, max_lon, max_lat = bbox

	x_min, y_max = lonlat_to_tile_indices(min_lon, min_lat, zoom)
	x_max, y_min = lonlat_to_tile_indices(max_lon, max_lat, zoom)

	# Ensure proper ordering
	if x_min > x_max:
		x_min, x_max = x_max, x_min
	if y_min > y_max:
		y_min, y_max = y_max, y_min

	for x in range(x_min, x_max + 1):
		for y in range(y_min, y_max + 1):
			yield zoom, x, y


def build_global_index(
	zooms: List[int],
	bbox: Optional[Tuple[float, float, float, float]] = None,
) -> List[Dict[str, object]]:
	"""
	Build a simple in-memory global tile index for given zoom levels.
	Returns a list of dicts with tile id and bounds.
	"""
	index: List[Dict[str, object]] = []
	for z in zooms:
		for zxy in iterate_tiles(z, bbox=bbox):
			_, x, y = zxy
			bounds = tile_indices_to_bounds(x, y, z)
			index.append({
				"id": tile_id(z, x, y),
				"z": z,
				"x": x,
				"y": y,
				"bounds": bounds,
			})
	return index


def key_from_latlon_time(lat: float, lon: float, year: int, month: int, zoom: int) -> str:
	"""
	Generate a stable key for a sample based on lat/lon/time and zoom tile.
	"""
	x, y = lonlat_to_tile_indices(lon, lat, zoom)
	return f"{tile_id(zoom, x, y)}/t{year:04d}{month:02d}"

