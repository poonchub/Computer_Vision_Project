tck, u = splprep(polygon_points.T, s=0, per=True)
    # u_new = np.linspace(u.min(), u.max(), 1000)  # เพิ่มจำนวนจุดเป็น 500
    # polygon_points = np.array(splev(u_new, tck)).T