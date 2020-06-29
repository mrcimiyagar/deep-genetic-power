import subprocess

rmse = float(subprocess.check_call(["python deep-learning-train.py " + str(114) + " " + str(34) + " " +
                                  str(125) + " " + str(0.01) + " " + str(100) + " " + str(36) + " " +
                                  str(67) + " " + str(36) + " " + str(78) + " " + str(32) + " " +
                                  str(1000)],
                                 shell=True))