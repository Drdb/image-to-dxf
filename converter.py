"""
Bitmap to DXF Converter - Core Engine
Converts bitmap images to DXF format with multiple modes:
- Threshold (lines)
- Floyd-Steinberg Dithering (dots)
- Outline (contours) - with improved sub-pixel interpolation and curvature-adaptive simplification
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
            smoothing_amount: Smoothing/simplification for outline mode (higher = fewer nodes)
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
            # Improved outline mode with sub-pixel interpolation
            num_levels = outline_levels
            
            if invert:
                img = Image.eval(img, lambda x: 255 - x)
            
            all_polylines = []
            
            if num_levels == 2:
                thresholds = [128]
            else:
                step = 256 // num_levels
                thresholds = [step * i for i in range(1, num_levels)]
            
            for thresh in thresholds:
                # Use improved sub-pixel marching squares
                edges = self._find_contours_subpixel(img, thresh)
                polylines = self._trace_contours(edges)
                
                processed = []
                for poly in polylines:
                    if len(poly) >= 2:
                        # Apply anti-staircase smoothing first
                        smoothed = self._smooth_staircase(poly, smoothing_amount)
                        
                        # Then apply curvature-adaptive simplification
                        simplified = self._simplify_adaptive(smoothed, smoothing_amount, corner_threshold)
                        
                        if len(simplified) >= 2:
                            processed.append(simplified)
                
                all_polylines.extend(processed)
            
            # Optimize path
            connection_tolerance = max(2.0, smoothing_amount)
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
    
    def _find_contours_subpixel(self, img, threshold):
        """
        Find contour edges using marching squares with sub-pixel interpolation.
        This produces smooth edges by interpolating the exact position based on pixel values.
        """
        w, h = img.size
        px = img.load()
        edges = []
        
        def interpolate(v1, v2, t):
            """Interpolate position between two pixels based on threshold."""
            if v1 == v2:
                return 0.5
            return (t - v1) / (v2 - v1)
        
        for y in range(h - 1):
            for x in range(w - 1):
                # Get pixel values
                val_tl = px[x, y]
                val_tr = px[x + 1, y]
                val_br = px[x + 1, y + 1]
                val_bl = px[x, y + 1]
                
                # Determine which corners are above threshold
                tl = 1 if val_tl >= threshold else 0
                tr = 1 if val_tr >= threshold else 0
                br = 1 if val_br >= threshold else 0
                bl = 1 if val_bl >= threshold else 0
                
                cell = tl * 8 + tr * 4 + br * 2 + bl * 1
                
                if cell == 0 or cell == 15:
                    continue
                
                # Calculate interpolated edge positions
                # Top edge (between tl and tr)
                t_top = interpolate(val_tl, val_tr, threshold)
                top = (x + t_top, y)
                
                # Right edge (between tr and br)
                t_right = interpolate(val_tr, val_br, threshold)
                right = (x + 1, y + t_right)
                
                # Bottom edge (between bl and br)
                t_bottom = interpolate(val_bl, val_br, threshold)
                bottom = (x + t_bottom, y + 1)
                
                # Left edge (between tl and bl)
                t_left = interpolate(val_tl, val_bl, threshold)
                left = (x, y + t_left)
                
                # Generate edges based on cell configuration
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
        
        # Build adjacency with tolerance for floating point coordinates
        def round_point(p, decimals=4):
            return (round(p[0], decimals), round(p[1], decimals))
        
        adjacency = {}
        for (p1, p2) in edges:
            rp1 = round_point(p1)
            rp2 = round_point(p2)
            if rp1 not in adjacency:
                adjacency[rp1] = []
            if rp2 not in adjacency:
                adjacency[rp2] = []
            adjacency[rp1].append((rp2, p1, p2))
            adjacency[rp2].append((rp1, p2, p1))
        
        used_edges = set()
        polylines = []
        
        def edge_key(p1, p2):
            return tuple(sorted([p1, p2]))
        
        for start_point in adjacency:
            for next_info in adjacency[start_point]:
                next_point, orig_start, orig_next = next_info
                ek = edge_key(start_point, next_point)
                if ek in used_edges:
                    continue
                
                polyline = [orig_start, orig_next]
                used_edges.add(ek)
                
                current = next_point
                while True:
                    found_next = False
                    for neighbor_info in adjacency.get(current, []):
                        neighbor, _, orig_neighbor = neighbor_info
                        ek = edge_key(current, neighbor)
                        if ek not in used_edges:
                            used_edges.add(ek)
                            polyline.append(orig_neighbor)
                            current = neighbor
                            found_next = True
                            break
                    if not found_next:
                        break
                
                current = start_point
                while True:
                    found_next = False
                    for neighbor_info in adjacency.get(current, []):
                        neighbor, orig_neighbor, _ = neighbor_info
                        ek = edge_key(current, neighbor)
                        if ek not in used_edges:
                            used_edges.add(ek)
                            polyline.insert(0, orig_neighbor)
                            current = neighbor
                            found_next = True
                            break
                    if not found_next:
                        break
                
                if len(polyline) >= 2:
                    polylines.append(polyline)
        
        return polylines
    
    def _smooth_staircase(self, points, smoothing_amount):
        """
        Remove staircase artifacts from contour while preserving overall shape.
        Uses a moving average filter that's aware of the local direction.
        """
        if len(points) < 5 or smoothing_amount <= 0:
            return points
        
        # Determine window size based on smoothing amount
        window = max(3, min(9, int(smoothing_amount * 2) + 1))
        if window % 2 == 0:
            window += 1
        half_window = window // 2
        
        is_closed = self._points_equal(points[0], points[-1])
        n = len(points) - 1 if is_closed else len(points)
        
        smoothed = []
        
        for i in range(n):
            # Collect points in window
            window_points = []
            for j in range(-half_window, half_window + 1):
                if is_closed:
                    idx = (i + j) % n
                else:
                    idx = max(0, min(n - 1, i + j))
                window_points.append(points[idx])
            
            # Calculate local curvature to decide smoothing strength
            if len(window_points) >= 3:
                curvature = self._estimate_curvature(window_points)
                # Less smoothing on high curvature areas (curves), more on low curvature (staircases)
                adaptive_weight = 1.0 / (1.0 + curvature * 10)
            else:
                adaptive_weight = 1.0
            
            # Weighted average - original point gets more weight on curves
            orig_weight = 1.0 - adaptive_weight * 0.7
            avg_x = sum(p[0] for p in window_points) / len(window_points)
            avg_y = sum(p[1] for p in window_points) / len(window_points)
            
            new_x = points[i][0] * orig_weight + avg_x * (1 - orig_weight)
            new_y = points[i][1] * orig_weight + avg_y * (1 - orig_weight)
            
            smoothed.append((new_x, new_y))
        
        if is_closed:
            smoothed.append(smoothed[0])
        
        return smoothed
    
    def _estimate_curvature(self, points):
        """Estimate local curvature from a set of points."""
        if len(points) < 3:
            return 0
        
        # Use the middle three points
        mid = len(points) // 2
        p1 = points[max(0, mid - 1)]
        p2 = points[mid]
        p3 = points[min(len(points) - 1, mid + 1)]
        
        # Calculate vectors
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        len1 = math.sqrt(v1[0]**2 + v1[1]**2)
        len2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if len1 < 0.001 or len2 < 0.001:
            return 0
        
        # Cross product gives sine of angle
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        
        # Normalize by segment lengths
        curvature = abs(cross) / (len1 * len2)
        
        return curvature
    
    def _simplify_adaptive(self, points, smoothing_amount, corner_threshold):
        """
        Curvature-adaptive simplification.
        Uses smaller tolerance on curves (preserves detail) and larger on straight lines (fewer nodes).
        """
        if len(points) <= 2:
            return points
        
        is_closed = self._points_equal(points[0], points[-1])
        
        # Base tolerance scales with smoothing amount
        # Higher smoothing = more simplification = fewer nodes
        base_tolerance = 0.3 + smoothing_amount * 0.2
        
        # First, detect corners (sharp turns) that should always be preserved
        corners = self._detect_corners(points, corner_threshold)
        corner_set = set(corners)
        
        # Add start/end points to preserved set
        corner_set.add(0)
        corner_set.add(len(points) - 1)
        
        # Calculate curvature at each point
        curvatures = self._calculate_all_curvatures(points)
        
        # Adaptive simplification
        result = [points[0]]
        last_added = 0
        
        for i in range(1, len(points) - 1):
            # Always keep corners
            if i in corner_set:
                result.append(points[i])
                last_added = i
                continue
            
            # Calculate local curvature (average of nearby points)
            local_curvature = curvatures[i]
            
            # Adaptive tolerance: small on curves, large on straight lines
            # High curvature = small tolerance = keep more points
            # Low curvature = large tolerance = remove more points
            adaptive_tolerance = base_tolerance / (1.0 + local_curvature * 20)
            adaptive_tolerance = max(0.1, min(base_tolerance * 3, adaptive_tolerance))
            
            # Check if this point deviates enough from the line between last added and next corner
            # Find next corner or end
            next_anchor = len(points) - 1
            for j in range(i + 1, len(points)):
                if j in corner_set:
                    next_anchor = j
                    break
            
            # Calculate distance from line
            dist = self._point_line_distance(points[i], points[last_added], points[next_anchor])
            
            # Also consider distance traveled - don't skip too many points
            segment_length = self._distance(points[last_added], points[i])
            
            # Keep point if it deviates enough OR we've traveled far enough
            max_segment = 5.0 + (10.0 / (1.0 + local_curvature * 10))  # Shorter segments on curves
            
            if dist > adaptive_tolerance or segment_length > max_segment:
                result.append(points[i])
                last_added = i
        
        result.append(points[-1])
        
        # Close if original was closed
        if is_closed and not self._points_equal(result[0], result[-1]):
            result.append(result[0])
        
        return result
    
    def _calculate_all_curvatures(self, points):
        """Calculate curvature at each point."""
        n = len(points)
        curvatures = [0.0] * n
        
        for i in range(1, n - 1):
            p1 = points[i - 1]
            p2 = points[i]
            p3 = points[i + 1]
            
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            len1 = math.sqrt(v1[0]**2 + v1[1]**2)
            len2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if len1 > 0.001 and len2 > 0.001:
                cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
                curvatures[i] = cross / (len1 * len2)
        
        return curvatures
    
    def _detect_corners(self, points, angle_threshold):
        """Find indices of corner points (sharp turns)."""
        if len(points) < 3:
            return []
        
        corners = []
        is_closed = self._points_equal(points[0], points[-1])
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
            if angle < 180 - angle_threshold:
                corners.append(i)
        
        return corners
    
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
    
    def _distance(self, p1, p2):
        """Calculate distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _points_equal(self, p1, p2, tolerance=0.001):
        """Check if two points are equal within tolerance."""
        return abs(p1[0] - p2[0]) < tolerance and abs(p1[1] - p2[1]) < tolerance
    
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
