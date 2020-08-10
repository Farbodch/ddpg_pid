class PidEnv():
    def __init__(self, sample_rate=1, setpoint=50):
        self.sample_rate = sample_rate
        self.setpoint = setpoint
        self.error = self.setpoint
        self.proportional = self.setpoint
        self.integral = 0
        self.derivative = 0
        self.last_error = self.error
        self.currpoint = 0
        self.kp = 1
        self.ki = 1
        self.kd = 1

    def step(self, action):
        self.kp = action[0]
        self.ki = action[1]
        self.kd = action[2]

        self.proportional = self.kp * self.error
        self.integral += self.error * self.sample_rate
        self.derivative = self.kd * (self.error - self.last_error) / self.sample_rate

        curr_input = self.proportional + self.ki * self.integral + self.derivative

        self.last_error = self.error
        self.currpoint += curr_input
        self.error = self.setpoint - self.currpoint

        reward = -abs(self.error)
        if reward == 0:
            reward = 10
        return (self.proportional, self.ki * self.integral, self.derivative, self.error, self.setpoint), reward

    def reset(self):
        self.error = self.setpoint
        self.proportional = self.setpoint
        self.integral = 0
        self.derivative = 0
        self.last_error = self.error
        self.currpoint = 0
        self.kp = 1
        self.ki = 1
        self.kd = 1
        return (self.proportional, self.integral, self.derivative, self.error, self.setpoint)

    def render(self):
        print("Error: "+str(self.error))
        print("Proportional Term: "+str(self.proportional))
        print("Integral Term: "+str(self.integral))
        print("Derivative Term: "+str(self.derivative))


