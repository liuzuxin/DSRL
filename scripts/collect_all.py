from fsrl.utils.easy_exp import ExperimentGrid

if __name__ == "__main__":

    exp_name = "benchmark"
    runner = ExperimentGrid(log_name=exp_name)

    # task = [
    #     "SafetyCarCircle-v0",
    #     "SafetyAntRun-v0",
    #     "SafetyDroneRun-v0",
    #     "SafetyDroneCircle-v0",
    #     "SafetyAntCircle-v0",
    # ]

    # task = [
    #     "SafetyPointGoal1Gymnasium-v0",
    #     "SafetyPointGoal2Gymnasium-v0",
    #     "SafetyPointButton1Gymnasium-v0",
    #     "SafetyPointButton2Gymnasium-v0",
    #     "SafetyPointPush1Gymnasium-v0",
    #     "SafetyPointPush2Gymnasium-v0",
    #     "SafetyCarGoal1Gymnasium-v0",
    #     "SafetyCarGoal2Gymnasium-v0",
    #     "SafetyCarButton1Gymnasium-v0",
    #     "SafetyCarButton2Gymnasium-v0",
    #     "SafetyCarPush1Gymnasium-v0",
    #     "SafetyCarPush2Gymnasium-v0",
    # ]

    task = [
        "SafetyHalfCheetahVelocityGymnasium-v1", "SafetyHopperVelocityGymnasium-v1",
        "SafetySwimmerVelocityGymnasium-v1", "SafetyWalker2dVelocityGymnasium-v1",
        "SafetyAntVelocityGymnasium-v1", "SafetyHumanoidVelocityGymnasium-v1"
    ]

    # Do not write & to the end of the command, it will be added automatically.
    template = "nohup  python scripts/train_trpol_v0.py --task {}"

    train_instructions = runner.compose(template, [task])
    # print(train_instructions)
    runner.start(train_instructions, max_parallel=12)