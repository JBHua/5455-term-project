from pymcd.mcd import Calculate_MCD

# gender = "female"
gender = "male"
accent = "philippines"
ground_truth = "f557d16817532d17167053abf86b9901daeb4c29042f4d37fa42ec57fe8f94babb00b99a0c3d9de47e6f4a049152b3121e6782a41bb9a00411202a0862479380"


mcd_toolbox = Calculate_MCD(MCD_mode="plain")

ground_truth_file = "/home/yifanhua/cv_01/clips/" + ground_truth + ".mp3"
test_dir = "./audio_eval/" + accent + "_" + gender + '/'

# two inputs w.r.t. reference (ground-truth) and synthesized speeches, respectively
mcd_value = mcd_toolbox.calculate_mcd(ground_truth_file, test_dir + "t5.wav")
print(f'mcd_value between ground truth and t5_vanilla: {mcd_value}')

mcd_value = mcd_toolbox.calculate_mcd(ground_truth_file, test_dir + "fine_tuned.wav")
print(f'mcd_value between ground truth and fine-tuned t5: {mcd_value}')
