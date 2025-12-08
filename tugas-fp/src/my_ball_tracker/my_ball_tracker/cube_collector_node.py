if self.state == "SEARCH":
            # 1. Panic Stop (Terlalu Dekat)
            if dist_front < 0.25: # Jarak panik sedikit dinaikkan biar lebih aman
                self.get_logger().warn(f"Panic Stop! Dist: {dist_front:.2f}m")
                twist.linear.x = -0.2 # Mundur lebih cepat dikit
                twist.angular.z = 0.0
            
            # 2. Target Locked (Visual)
            elif target_visible:
                self.get_logger().info("Target Locked! Approaching...")
                self.state = "APPROACH"
                twist.linear.x = 0.0
                twist.angular.z = 0.0

            # 3. Target Lost Nearby (Reverse Recovery)
            elif dist_front < 0.50 and time_since_seen < 2.0:
                 self.get_logger().info("Target lost nearby! Reversing...")
                 twist.linear.x = -0.15
                 twist.angular.z = 0.0
                
            # 4. Obstacle Avoidance (Reference Style: Simple & Robust)
            # Menggunakan logika referensi: Jika ada halangan, putar satu arah (biasa kanan/kiri) sampai lolos.
            # Hindari membandingkan kiri vs kanan yang menyebabkan osilasi (stuck).
            elif dist_front < 0.75: 
                # self.get_logger().warn(f"Obstacle ({dist_front:.2f}m). Rotating...")
                twist.linear.x = 0.0
                # FIXED: Putar konstan ke Kanan (-z) atau Kiri (+z) tanpa ragu.
                # Sesuai referensi kamu: rotate_angle_velocity = -0.3 (Kanan)
                twist.angular.z = -0.5 
            
            # 5. Memory Recovery (Scan last position)
            elif time_since_seen < 3.0:
                self.get_logger().info("Scanning for lost target...")
                twist.linear.x = 0.0
                direction = 1.0 if self.last_known_error > 0 else -1.0
                twist.angular.z = direction * 0.6 

            # 6. Wander (Jalan-jalan)
            else:
                twist.linear.x = 0.25 
                twist.angular.z = 0.0 # Lurus saja kalau kosong, atau sedikit wobble 0.05