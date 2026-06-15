class AlertEngine:
    def __init__(self, capacity=200):
        self.capacity = capacity
        self.current_alerts = []

    def evaluate(self, inside_count, density_percentage, density_level):
        alerts = []
        
        # Capacity checks
        occupancy = (inside_count / self.capacity) * 100 if self.capacity > 0 else 0
        
        if occupancy >= 95:
            alerts.append({"level": "CRITICAL", "message": f"Critical Occupancy: {int(occupancy)}% Capacity Reached!"})
        elif occupancy >= 80:
            alerts.append({"level": "WARNING", "message": f"High Occupancy: {int(occupancy)}% Capacity Reached"})
            
        # Density checks
        if density_level == "CRITICAL":
            alerts.append({"level": "CRITICAL", "message": "Dangerous Crowd Density Detected!"})
        elif density_level == "HIGH":
            alerts.append({"level": "WARNING", "message": "High Crowd Density. Prepare crowd control."})
            
        self.current_alerts = alerts
        return alerts

    def get_alerts(self):
        return self.current_alerts
