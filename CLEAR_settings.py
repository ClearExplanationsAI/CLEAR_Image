from tkinter import *

""" Specifes CLEAR'S user input parameters. CLEAR sets the input parameters as global variables
whose values are NOT changed in any other module (these are CLEAR's only global variables).
Tkinter is used to provide some checks on the user's inputs. The file 
'Input Parameters for CLEAR.pdf' on Github documents the input parameters.
 """


def init():
    global case_study, max_predictors, num_samples, regression_type, \
        score_type, test_sample, CLEAR_path,input_images_path,images_to_process, \
        GAN_images_path, apply_counterfactual_weights, counterfactual_weight, \
        generate_regression_files, interactions_only, no_intercept, centering, no_polynomimals_no_interactions, \
        use_prev_sensitivity, binary_decision_boundary,include_features, \
        include_features_list, image_infill,image_all_segments,\
        image_segment_threshold, image_use_old_synthetic,image_segment_type,\
        debug_mode,image_model_name,image_classes, logistic_regularise, sufficiency_threshold,\
        max_sufficient_causes, image_counterfactual_interactions


    case_study = 'Medical' #'Medical' or 'Synthetic'
    max_predictors = 6# maximum number of dependent variables in stepwise regression  was 10 for Xray, 6 for synthetic
    num_samples = 1000  # number of observations to generate in Synthetic Dataset. Default 1000
    regression_type = 'logistic'  # 'multiple' 'logistic'
    logistic_regularise = False
    score_type = 'aic'  # prsquared is McFadden Pseudo R-squared. Can also be
    #                          set to aic or adjR (adjusted R-squared)
    CLEAR_path = 'D:/NeurIPS CLEAR/'  # e.g. 'D:/CLEAR/'
    # input_images_path = 'D:/Synthetic/synthetic_data/diseased/'
    # GAN_images_path = 'D:/Synthetic/synthetic_data/gen_healthy/'
    input_images_path = 'D:/NeurIPS CLEAR/CheXpert Data/1_pleural_effusion/'
    GAN_images_path = 'D:/NeurIPS CLEAR/CheXpert Data/0_healthy/'
    images_to_process = 'Medical Images Neurips.csv' # Synthetic Images Neurips.csv’,'Medical Images Neurips.csv'
    apply_counterfactual_weights = True
    counterfactual_weight = 200  # weighting applied to each counterfactual image in regression
    generate_regression_files = False
    binary_decision_boundary = 0.5
    # Parameters for evaluating the effects of different parts of CLEAR's regression
    no_polynomimals_no_interactions = True
    interactions_only = True
    no_intercept = False # only for multiple regression - probably delete this, it was for early version of CLEAR IMAGE - Otherwise need to change Adjusted R2
    centering = True #forces CLEAR's regression to pass through observation that is to be explained.
    # Parameters for forcing features to be included in regression
    include_features = False # Features in 'include_feature_list' will be forced into regression equation
    include_features_list = []
    debug_mode = False
    sufficiency_threshold = 0.99
    max_sufficient_causes = 2 # This should be set to 1 or 2
    image_model_name= 'Densenet_Neurips_CheXpert.pth' #'VGG_Neurips_Synthetic.pth 'Densenet_Neurips_Synthetic.pth'
     #'Densenet_Neurips_CheXpert.pth','VGG16_Neurips_CheXpert.pth'
    image_infill ='GAN' # 'GAN', 'average','none','black'
    image_all_segments= False
    image_segment_threshold = 0.05
    image_use_old_synthetic = False  # Only set to True when testing code
    image_counterfactual_interactions = False
    image_segment_type ='GAN_Xrays'  #'GAN_Xrays','felzenszwalb','GAN_diff','quickshift'
    #inputs specifically for X-rays with GAN images
    image_classes =['normal','effusion'] #['normal','diseased'],['normal','effusion']
    check_input_parameters()
""" Check if input parameters are consistent"""


def check_input_parameters():
    def close_program():
        root.destroy()
        sys.exit()

    error_msg = ""
    if regression_type == 'logistic' and \
            (score_type != 'prsquared' and score_type != 'aic'):
        error_msg = "logistic regression and score type combination incorrectly specified"
    elif regression_type == 'multiple' and score_type == 'prsquared':
        error_msg = "McFadden Pseudo R-squared cannot be used with multiple regression"
    elif no_intercept == True and regression_type != 'multiple':
        error_msg = "no intercept requires regression type to be multiple"
    elif no_intercept == True and centering == True:
        error_msg = "centering requires no-intercept to be False"
    elif case_study not in ['Medical', 'Synthetic']:
        error_msg = "case study incorrectly specified"
    elif regression_type not in ['multiple', 'logistic']:
        error_msg = "Regression type misspecified"
    elif (isinstance((interactions_only & centering & no_polynomimals_no_interactions &
                      apply_counterfactual_weights & generate_regression_files), bool)) is False:
        error_msg = "A boolean variable has been incorrectly specified"

    if error_msg != "":
        root = Tk()
        root.title("Input Error in CLEAR_settings")
        root.geometry("350x150")

        label_1 = Label(root, text=error_msg,
                        justify=CENTER, height=4, wraplength=150)
        button_1 = Button(root, text="OK",
                          padx=5, pady=5, command=close_program)
        label_1.pack()
        button_1.pack()
        root.attributes("-topmost", True)
        root.focus_force()
        root.mainloop()

#['LIMITBAL', 'AGE', 'PAY0', 'PAY6', 'BILLAMT1', 'BILLAMT6', 'PAYAMT1', 'PAYAMT6','MARRIAGE']
