"""
Bitmap to DXF Converter - Core Engine
Converts bitmap images to DXF format with multiple modes:
- Threshold (lines)
- Floyd-Steinberg Dithering (dots)
- Outline (contours)
"""

from PIL import Image
from pathlib import Path
import math
import io


class BitmapToDXFConverter:
    """Core conversion engine for bitmap to DXF conversion."""
    
    def __init__(self):
        pass
    
    def convert(
        self,
        input_image,
        mode="threshold",
        image_height_um=1000.0,
        spot_size_um=5.0,
        spot_spacing_factor=1.1,
        threshold=200,
        invert=False,
        flip_y=True,
        bidirectional=True,
        outline_levels=2,
        smoothing_amount=2.0,
        corner_threshold=45.0
    ):
        """
        Convert an image to DXF format.
        
        Args:
            input_image: PIL Image or file path or bytes
            mode: "threshold", "floyd_steinberg", or "outline"
            image_height_um: Target height in microns
            spot_size_um: Laser spot size in microns
            spot_spacing_factor: Spacing between spots (multiplier)
            threshold: Threshold value for threshold mode (0-255)
            invert: Swap black/white
            flip_y: Flip Y axis for CAD orientation
            bidirectional: Use zigzag scanning
            outline_levels: Number of grayscale levels for outline mode
            smoothing_amount: Smoothing for outline mode
            corner_threshold: Corner detection angle for outline mode
            
        Returns:
            tuple: (dxf_content_string, stats_dict)
        """
        # Load image
        if isinstance(input_image, (str, Path)):
            img = Image.open(input_image).convert("L")
        elif isinstance(input_image, bytes):
            img = Image.open(io.BytesIO(input_image)).convert("L")
        elif isinstance(input_image, Image.Image):
            img = input_image.convert("L")
        else:
            raise ValueError("input_image must be a file path, bytes, or PIL Image")
        
        w, h = img.size
        
        # Calculate dimensions
        aspect_ratio = w / h
        image_width_um = image_height_um * aspect_ratio
        
        pixel_size_x = image_width_um / w
        pixel_size_y = image_height_um / h
        
        # Calculate line step based on spot size
        min_spacing = spot_size_um * spot_spacing_factor
        line_step = max(1, int(math.ceil(min_spacing / pixel_size_y)))
        
        # Apply dithering if needed
        if mode == "floyd_steinberg":
            if invert:
                img = Image.eval(img, lambda x: 255 - x)
            img = self._apply_floyd_steinberg_dithering(img)
        
        px = img.load()
        
        def is_black(v):
            if mode == "floyd_steinberg":
                return v < 128
            else:
                vb = v < threshold
                return (not vb) if invert else vb
        
        # Build DXF content
        dxf_lines = []
        
        # Header section (minimal DXF R12)
        dxf_lines.extend(["0", "SECTION", "2", "HEADER"])
        dxf_lines.extend(["9", "$ACADVER", "1", "AC1009"])
        dxf_lines.extend(["0", "ENDSEC"])
        
        # Entities section
        dxf_lines.extend(["0", "SECTION", "2", "ENTITIES"])
        
        entity_count = 0
        
        if mode == "floyd_steinberg":
            # Dithering mode: create tiny LINE segments for each dot
            dot_length = 1
            
            for y in range(0, h, line_step):
                row_num = y // line_step
                reverse = bidirectional and (row_num % 2 == 1)
                
                x_positions = range(0, w, line_step)
                if reverse:
                    x_positions = reversed(list(x_positions))
                
                for x in x_positions:
                    if is_black(px[x, y]):
                        X = round(x * pixel_size_x, 1)
                        Y = round((h - 1 - y if flip_y else y) * pixel_size_y, 1)
                        
                        dxf_lines.extend([
                            "0", "LINE",
                            "8", "0",
                            "10", f"{X:.1f}",
                            "20", f"{Y:.1f}",
                            "11", f"{X + dot_length:.1f}",
                            "21", f"{Y:.1f}"
                        ])
                        entity_count += 1
                        
        elif mode == "outline":
            # Outline mode: trace contours
            num_levels = outline_levels
            smoothing = smoothing_amount
            corner_angle = corner_threshold
            
            if invert:
                img = Image.eval(img, lambda x: 255 - x)
            
            all_polylines = []
            
            if num_levels == 2:
                thresholds = [128]
            else:
                step = 256 // num_levels
                thresholds = [step * i for i in range(1, num_levels)]
            
            for thresh in thresholds:
                edges = self._find_contours_marching_squares(img, thresh)
                polylines = self._trace_contours(edges)
                
                processed = []
                for poly in polylines:
                    if len(poly) >= 2:
                        if smoothing > 0:
                            smoothed = self._smooth_contour(poly, smoothing, corner_angle)
                        else:
                            smoothed = poly
                        
                        simp = self._simplify_polyline(smoothed, smoothing * 0.5 + 0.5)
                        if len(simp) >= 2:
                            processed.append(simp)
                
                all_polylines.extend(processed)
            
            # Optimize path
            connection_tolerance = max(2.0, smoothing)
            optimized = self._optimize_outline_path(all_polylines, connection_tolerance)
            
            # Write polylines as LINE segments
            for poly in optimized:
                for i in range(len(poly) - 1):
                    x1, y1 = poly[i]
                    x2, y2 = poly[i + 1]
                    
                    X1 = round(x1 * pixel_size_x, 1)
                    Y1 = round((h - 1 - y1 if flip_y else y1) * pixel_size_y, 1)
                    X2 = round(x2 * pixel_size_x, 1)
                    Y2 = round((h - 1 - y2 if flip_y else y2) * pixel_size_y, 1)
                    
                    dxf_lines.extend([
                        "0", "LINE",
                        "8", "0",
                        "10", f"{X1:.1f}",
                        "20", f"{Y1:.1f}",
                        "11", f"{X2:.1f}",
                        "21", f"{Y2:.1f}"
                    ])
                    entity_count += 1
                    
        else:  # threshold mode
            # Create LINE segments for each run
            for y in range(0, h, line_step):
                row_num = y // line_step
                reverse = bidirectional and (row_num % 2 == 1)
                
                x = 0
                row_segments = []
                while x < w:
                    while x < w and not is_black(px[x, y]):
                        x += 1
                    if x >= w:
                        break
                    x1 = x
                    while x < w and is_black(px[x, y]):
                        x += 1
                    x2 = x - 1
                    row_segments.append((x1, x2))
                
                Y = round((h - 1 - y if flip_y else y) * pixel_size_y, 1)
                
                if reverse:
                    for x1, x2 in reversed(row_segments):
                        X1 = round((x2 + 1) * pixel_size_x, 1)
                        X2 = round(x1 * pixel_size_x, 1)
                        dxf_lines.extend([
                            "0", "LINE",
                            "8", "0",
                            "10", f"{X1:.1f}",
                            "20", f"{Y:.1f}",
                            "11", f"{X2:.1f}",
                            "21", f"{Y:.1f}"
                        ])
                        entity_count += 1
                else:
                    for x1, x2 in row_segments:
                        X1 = round(x1 * pixel_size_x, 1)
                        X2 = round((x2 + 1) * pixel_size_x, 1)
                        dxf_lines.extend([
                            "0", "LINE",
                            "8", "0",
                            "10", f"{X1:.1f}",
                            "20", f"{Y:.1f}",
                            "11", f"{X2:.1f}",
                            "21", f"{Y:.1f}"
                        ])
                        entity_count += 1
        
        # End file
        dxf_lines.extend(["0", "ENDSEC", "0", "EOF"])
        
        dxf_content = "\n".join(dxf_lines)
        
        stats = {
            "width_px": w,
            "height_px": h,
            "width_um": image_width_um,
            "height_um": image_height_um,
            "entity_count": entity_count,
            "mode": mode,
            "line_step": line_step,
            "file_size_kb": len(dxf_content) / 1024
        }
        
        return dxf_content, stats
    
    def _apply_floyd_steinberg_dithering(self, img):
        """Apply Floyd-Steinberg dithering to grayscale image."""
        w, h = img.size
        pixels = list(img.getdata())
        
        data = [[float(pixels[y * w + x]) for x in range(w)] for y in range(h)]
        
        for y in range(h):
            for x in range(w):
                old_pixel = data[y][x]
                new_pixel = 255 if old_pixel > 127 else 0
                data[y][x] = new_pixel
                error = old_pixel - new_pixel
                
                if x + 1 < w:
                    data[y][x + 1] += error * 7 / 16
                if y + 1 < h:
                    if x > 0:
                        data[y + 1][x - 1] += error * 3 / 16
                    data[y + 1][x] += error * 5 / 16
                    if x + 1 < w:
                        data[y + 1][x + 1] += error * 1 / 16
        
        result = Image.new('L', (w, h))
        result_data = [int(max(0, min(255, data[y][x]))) for y in range(h) for x in range(w)]
        result.putdata(result_data)
        return result
    
    def _find_contours_marching_squares(self, img, threshold):
        """Find contour edges using marching squares algorithm."""
        w, h = img.size
        px = img.load()
        edges = []
        
        for y in range(h - 1):
            for x in range(w - 1):
                tl = 1 if px[x, y] >= threshold else 0
                tr = 1 if px[x + 1, y] >= threshold else 0
                br = 1 if px[x + 1, y + 1] >= threshold else 0
                bl = 1 if px[x, y + 1] >= threshold else 0
                
                cell = tl * 8 + tr * 4 + br * 2 + bl * 1
                
                if cell == 0 or cell == 15:
                    continue
                
                top = (x + 0.5, y)
                right = (x + 1, y + 0.5)
                bottom = (x + 0.5, y + 1)
                left = (x, y + 0.5)
                
                if cell == 1 or cell == 14:
                    edges.append((left, bottom))
                elif cell == 2 or cell == 13:
                    edges.append((bottom, right))
                elif cell == 3 or cell == 12:
                    edges.append((left, right))
                elif cell == 4 or cell == 11:
                    edges.append((top, right))
                elif cell == 5:
                    edges.append((left, top))
                    edges.append((bottom, right))
                elif cell == 6 or cell == 9:
                    edges.append((top, bottom))
                elif cell == 7 or cell == 8:
                    edges.append((left, top))
                elif cell == 10:
                    edges.append((top, right))
                    edges.append((left, bottom))
        
        return edges
    
    def _trace_contours(self, edges):
        """Connect edge segments into continuous polylines."""
        if not edges:
            return []
        
        adjacency = {}
        for (p1, p2) in edges:
            if p1 not in adjacency:
                adjacency[p1] = []
            if p2 not in adjacency:
                adjacency[p2] = []
            adjacency[p1].append(p2)
            adjacency[p2].append(p1)
        
        used_edges = set()
        polylines = []
        
        def edge_key(p1, p2):
            return tuple(sorted([p1, p2]))
        
        for start_point in adjacency:
            for next_point in adjacency[start_point]:
                ek = edge_key(start_point, next_point)
                if ek in used_edges:
                    continue
                
                polyline = [start_point, next_point]
                used_edges.add(ek)
                
                current = next_point
                while True:
                    found_next = False
                    for neighbor in adjacency.get(current, []):
                        ek = edge_key(current, neighbor)
                        if ek not in used_edges:
                            used_edges.add(ek)
                            polyline.append(neighbor)
                            current = neighbor
                            found_next = True
                            break
                    if not found_next:
                        break
                
                current = start_point
                while True:
                    found_next = False
                    for neighbor in adjacency.get(current, []):
                        ek = edge_key(current, neighbor)
                        if ek not in used_edges:
                            used_edges.add(ek)
                            polyline.insert(0, neighbor)
                            current = neighbor
                            found_next = True
                            break
                    if not found_next:
                        break
                
                if len(polyline) >= 2:
                    polylines.append(polyline)
        
        return polylines
    
    def _simplify_polyline(self, points, tolerance):
        """Douglas-Peucker polyline simplification."""
        if len(points) <= 2:
            return points
        
        is_closed = (points[0] == points[-1])
        
        if is_closed and len(points) <= 4:
            return points
        
        if is_closed:
            max_dist = 0
            split_idx = 1
            
            for i in range(1, len(points) - 1):
                prev_pt = points[i - 1]
                next_pt = points[(i + 1) % (len(points) - 1)]
                dist = self._point_line_distance(points[i], prev_pt, next_pt)
                if dist > max_dist:
                    max_dist = dist
                    split_idx = i
            
            reordered = points[split_idx:-1] + points[:split_idx + 1]
            simplified = self._simplify_open(reordered, tolerance)
            
            if simplified[0] != simplified[-1]:
                simplified.append(simplified[0])
            
            return simplified
        else:
            return self._simplify_open(points, tolerance)
    
    def _simplify_open(self, points, tolerance):
        """Douglas-Peucker simplification for open polylines."""
        if len(points) <= 2:
            return points
        
        first = points[0]
        last = points[-1]
        
        max_dist = 0
        max_idx = 0
        
        for i in range(1, len(points) - 1):
            dist = self._point_line_distance(points[i], first, last)
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        
        if max_dist > tolerance:
            left = self._simplify_open(points[:max_idx + 1], tolerance)
            right = self._simplify_open(points[max_idx:], tolerance)
            return left[:-1] + right
        else:
            return [first, last]
    
    def _point_line_distance(self, point, line_start, line_end):
        """Calculate perpendicular distance from point to line segment."""
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return math.sqrt((x - x1)**2 + (y - y1)**2)
        
        t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)))
        
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        
        return math.sqrt((x - proj_x)**2 + (y - proj_y)**2)
    
    def _calculate_angle(self, p1, p2, p3):
        """Calculate the angle at p2 formed by p1-p2-p3."""
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        len1 = math.sqrt(v1[0]**2 + v1[1]**2)
        len2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if len1 == 0 or len2 == 0:
            return 180.0
        
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        cos_angle = max(-1, min(1, dot / (len1 * len2)))
        
        return math.degrees(math.acos(cos_angle))
    
    def _detect_corners(self, points, angle_threshold):
        """Find indices of corner points."""
        if len(points) < 3:
            return []
        
        corners = []
        is_closed = (points[0] == points[-1])
        n = len(points) - 1 if is_closed else len(points)
        
        for i in range(1, n):
            if is_closed:
                p1 = points[(i - 1) % n]
                p2 = points[i % n]
                p3 = points[(i + 1) % n]
            else:
                if i == 0 or i == len(points) - 1:
                    continue
                p1 = points[i - 1]
                p2 = points[i]
                p3 = points[i + 1]
            
            angle = self._calculate_angle(p1, p2, p3)
            
            if angle < angle_threshold:
                corners.append(i)
        
        return corners
    
    def _gaussian_weight(self, distance, sigma):
        """Calculate Gaussian weight."""
        if sigma <= 0:
            return 1.0 if distance == 0 else 0.0
        return math.exp(-(distance**2) / (2 * sigma**2))
    
    def _smooth_segment(self, points, sigma):
        """Apply Gaussian smoothing to a segment."""
        if len(points) <= 2 or sigma <= 0:
            return points
        
        smoothed = []
        window = int(sigma * 3)
        
        for i in range(len(points)):
            sum_x = 0
            sum_y = 0
            sum_weight = 0
            
            for j in range(max(0, i - window), min(len(points), i + window + 1)):
                weight = self._gaussian_weight(abs(j - i), sigma)
                sum_x += points[j][0] * weight
                sum_y += points[j][1] * weight
                sum_weight += weight
            
            if sum_weight > 0:
                smoothed.append((sum_x / sum_weight, sum_y / sum_weight))
            else:
                smoothed.append(points[i])
        
        return smoothed
    
    def _smooth_contour(self, points, smoothing_pixels, corner_angle_threshold):
        """Smooth a contour while preserving sharp corners."""
        if len(points) < 3 or smoothing_pixels <= 0:
            return points
        
        is_closed = (points[0] == points[-1])
        corners = self._detect_corners(points, corner_angle_threshold)
        
        if not corners:
            if is_closed:
                extended = points[:-1] + points[:-1] + points[:-1]
                n = len(points) - 1
                smoothed_ext = self._smooth_segment(extended, smoothing_pixels)
                smoothed = smoothed_ext[n:2*n]
                smoothed.append(smoothed[0])
                return smoothed
            else:
                return self._smooth_segment(points, smoothing_pixels)
        
        if is_closed:
            all_corners = sorted(set(corners))
        else:
            all_corners = sorted(set([0] + corners + [len(points) - 1]))
        
        smoothed_points = []
        
        if is_closed:
            n_corners = len(all_corners)
            for i in range(n_corners):
                start_idx = all_corners[i]
                end_idx = all_corners[(i + 1) % n_corners]
                
                if end_idx > start_idx:
                    segment = points[start_idx:end_idx + 1]
                else:
                    segment = points[start_idx:-1] + points[:end_idx + 1]
                
                if len(segment) > 2:
                    middle = self._smooth_segment(segment, smoothing_pixels)
                    middle[0] = segment[0]
                    middle[-1] = segment[-1]
                    smoothed_points.extend(middle[:-1])
                else:
                    smoothed_points.extend(segment[:-1])
            
            smoothed_points.append(smoothed_points[0])
        else:
            for i in range(len(all_corners) - 1):
                start_idx = all_corners[i]
                end_idx = all_corners[i + 1]
                
                segment = points[start_idx:end_idx + 1]
                
                if len(segment) > 2:
                    middle = self._smooth_segment(segment, smoothing_pixels)
                    middle[0] = segment[0]
                    middle[-1] = segment[-1]
                    
                    if i == 0:
                        smoothed_points.extend(middle)
                    else:
                        smoothed_points.extend(middle[1:])
                else:
                    if i == 0:
                        smoothed_points.extend(segment)
                    else:
                        smoothed_points.extend(segment[1:])
        
        return smoothed_points
    
    def _optimize_outline_path(self, polylines, connection_tolerance=2.0):
        """Optimize outline path for continuous laser travel."""
        if not polylines or len(polylines) <= 1:
            return polylines
        
        def distance(p1, p2):
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        def get_endpoints(poly):
            return poly[0], poly[-1]
        
        n = len(polylines)
        endpoints = [get_endpoints(p) for p in polylines]
        
        connections = [[] for _ in range(n)]
        
        for i in range(n):
            start_i, end_i = endpoints[i]
            for j in range(n):
                if i == j:
                    continue
                start_j, end_j = endpoints[j]
                
                d_ss = distance(start_i, start_j)
                d_se = distance(start_i, end_j)
                d_es = distance(end_i, start_j)
                d_ee = distance(end_i, end_j)
                
                if d_ss <= connection_tolerance:
                    connections[i].append((j, 0, 0, d_ss))
                if d_se <= connection_tolerance:
                    connections[i].append((j, 0, 1, d_se))
                if d_es <= connection_tolerance:
                    connections[i].append((j, 1, 0, d_es))
                if d_ee <= connection_tolerance:
                    connections[i].append((j, 1, 1, d_ee))
        
        for i in range(n):
            connections[i].sort(key=lambda x: x[3])
        
        used = [False] * n
        chains = []
        
        def build_chain(start_idx):
            chain = [(start_idx, False)]
            used[start_idx] = True
            
            while True:
                last_idx, last_reversed = chain[-1]
                last_poly = polylines[last_idx]
                if last_reversed:
                    current_endpoint = 0
                else:
                    current_endpoint = 1
                
                best_next = None
                best_dist = float('inf')
                best_reverse = False
                
                for other_idx, my_ep, their_ep, dist in connections[last_idx]:
                    if used[other_idx]:
                        continue
                    if last_reversed:
                        my_actual_ep = 1 - my_ep
                    else:
                        my_actual_ep = my_ep
                    
                    if my_actual_ep == 1:
                        if dist < best_dist:
                            best_dist = dist
                            best_next = other_idx
                            best_reverse = (their_ep == 1)
                
                if best_next is not None:
                    chain.append((best_next, best_reverse))
                    used[best_next] = True
                else:
                    break
            
            while True:
                first_idx, first_reversed = chain[0]
                if first_reversed:
                    current_endpoint = 1
                else:
                    current_endpoint = 0
                
                best_prev = None
                best_dist = float('inf')
                best_reverse = False
                
                for other_idx, my_ep, their_ep, dist in connections[first_idx]:
                    if used[other_idx]:
                        continue
                    if first_reversed:
                        my_actual_ep = 1 - my_ep
                    else:
                        my_actual_ep = my_ep
                    
                    if my_actual_ep == 0:
                        if dist < best_dist:
                            best_dist = dist
                            best_prev = other_idx
                            best_reverse = (their_ep == 0)
                
                if best_prev is not None:
                    chain.insert(0, (best_prev, best_reverse))
                    used[best_prev] = True
                else:
                    break
            
            return chain
        
        for i in range(n):
            if not used[i]:
                chain = build_chain(i)
                chains.append(chain)
        
        def chain_endpoints(chain):
            first_idx, first_rev = chain[0]
            last_idx, last_rev = chain[-1]
            
            first_poly = polylines[first_idx]
            last_poly = polylines[last_idx]
            
            if first_rev:
                chain_start = first_poly[-1]
            else:
                chain_start = first_poly[0]
            
            if last_rev:
                chain_end = last_poly[0]
            else:
                chain_end = last_poly[-1]
            
            return chain_start, chain_end
        
        current_pos = (0, 0)
        remaining_chains = list(range(len(chains)))
        ordered_chains = []
        
        while remaining_chains:
            best_chain_idx = None
            best_dist = float('inf')
            best_reverse_chain = False
            
            for ci in remaining_chains:
                chain_start, chain_end = chain_endpoints(chains[ci])
                
                dist_to_start = distance(current_pos, chain_start)
                dist_to_end = distance(current_pos, chain_end)
                
                if dist_to_start < best_dist:
                    best_dist = dist_to_start
                    best_chain_idx = ci
                    best_reverse_chain = False
                
                if dist_to_end < best_dist:
                    best_dist = dist_to_end
                    best_chain_idx = ci
                    best_reverse_chain = True
            
            chain = chains[best_chain_idx]
            if best_reverse_chain:
                chain = [(idx, not rev) for idx, rev in reversed(chain)]
            
            ordered_chains.append(chain)
            
            _, chain_end = chain_endpoints(chain)
            current_pos = chain_end
            remaining_chains.remove(best_chain_idx)
        
        result = []
        for chain in ordered_chains:
            for poly_idx, reversed_flag in chain:
                poly = polylines[poly_idx]
                if reversed_flag:
                    poly = list(reversed(poly))
                result.append(poly)
        
        return result
