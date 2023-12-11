from pymcd.mcd import Calculate_MCD

gender = "female"
accent = "indian"
ground_truth = "b0ad186dd1d0187426a1b546db6ea0a43b11dd9957a4579016f528850af60b98c8c7f8761be1e418c56609897972b13892c33f3a0b74420d5f745dbca1b7b000"


mcd_toolbox = Calculate_MCD(MCD_mode="plain")

ground_truth_file = "/home/yifanhua/cv_01/clips/" + ground_truth + ".mp3"
test_dir = "./audio_eval/" + accent + "_" + gender + '/'

# two inputs w.r.t. reference (ground-truth) and synthesized speeches, respectively
mcd_value = mcd_toolbox.calculate_mcd(ground_truth_file, test_dir + "t5.wav")
print(f'mcd_value between ground truth and t5_vanilla for us male: {mcd_value}')

mcd_value = mcd_toolbox.calculate_mcd(ground_truth_file, test_dir + "fine_tuned.wav")
print(f'mcd_value between ground truth and fine-tuned t5 for us male: {mcd_value}')
