
def build_horizontal_rule(cols):
	row_str = "+"
	for c in range(0, cols):
		row_str += "-"*20 + "+"
	row_str += "\n"
	return row_str

def build_vertical_lines(cols):
	row_str = "|"
	for c in range(0, cols):
		row_str +=" "*20 + "|"
	row_str += "\n"
	return row_str

def draw_grid(rows, cols):
	hr = build_horizontal_rule(cols)
	vr = build_vertical_lines(cols)
	grid = hr
	for r in range(0, rows):
		for i in range(0,7):
			grid += vr
		grid += hr
	return grid

v = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]



def draw_grid_with_V_values(self, rows, cols):
	hr = build_horizontal_rule(cols)
	grid_str = hr
	for r in range(0, rows):
		for height in range(0, 7):
			grid_str +="|"
			for c in range(0, cols):
				if height == 3:
					next_num = str(round(v.pop(), 3))
					num_width = len(next_num)
					left_padding = math.ceil((20-num_width) / 2)
					right_padding = math.floor((20-num_width) / 2)
					grid_str += " "*left_padding + next_num + " "*right_padding
				else:
					grid_str += " "*20
				grid_str += "|"
			grid_str += "\n"
		grid_str += hr
	return grid_str



