# Outstanding - Calculate_Perturbations() does not allow for categorical multi-class
#                                       create missing_log_df function
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from math import log10, floor, log, exp
from sympy import symbols, solve, simplify
from skimage.segmentation import mark_boundaries
# from keras.preprocessing.image import save_img
import cv2
import io
from PIL import Image

import CLEAR_settings, CLEAR_regression


class CLEARPerturbation(object):
    # Contains features specific to a particular b-perturbation
    def __init__(self):
        self.wTx = None
        self.nncomp_idx = None
        self.target_feature = None
        self.obs = None
        self.newnn_class = None
        self.raw_weights = None
        self.raw_eqn = None
        self.raw_data = None
        self.adj_raw_data = None
        self.target_prob = None

def getSufficientCauses(wPerturb):
    sufficient_causes = []
    intercept_negWeights = wPerturb.intercept
    for i in wPerturb.raw_weights:
        if i <0:
            intercept_negWeights += i
    for idx, val in enumerate(wPerturb.raw_weights):
        #From the logistics equation ln(y/(1-y)) = wTx
        if (val + intercept_negWeights) > np.log(CLEAR_settings.sufficiency_threshold/(1-CLEAR_settings.sufficiency_threshold)):
            sufficient_causes.append(wPerturb.raw_eqn[idx])
    if len(sufficient_causes) == 0 and CLEAR_settings.max_sufficient_causes ==2:
        for j in range (0,len(wPerturb.raw_weights-1)):
            for k in range (j+1,len(wPerturb.raw_weights) ):
                if intercept_negWeights +  wPerturb.raw_weights[j] + wPerturb.raw_weights[k] > \
                        np.log(CLEAR_settings.sufficiency_threshold/(1-CLEAR_settings.sufficiency_threshold)):
                    sufficient_causes.append([wPerturb.raw_eqn[j],wPerturb.raw_eqn[k]])
    return(sufficient_causes)

def Calculate_Perturbations(explainer, results_df):
    """ b-perturbations are now calculated and stored
        in the nncomp_df dataframe. If CLEAR calculates a b-perturbation
        that is infeasible, then the details of the b-perturbation
        are stated in the missing_log_df dataframe. CLEAR will classify
        a b-perturbation as being infeasible if it is outside 'the feasibility
        range' it calculates for each feature.
    """
    print("\n Calculating b-counterfactuals \n")
    nncomp_df = pd.DataFrame(columns=['observation', 'feature', 'orgFeatValue', 'orgAiProb',
                                      'actPerturbedFeatValue', 'AiProbWithActPerturbation', 'estPerturbedFeatValue',
                                      'errorPerturbation', 'regProbWithActPerturbation',
                                      'errorRegProbActPerturb', 'orgClass','sufficient'])
    wPerturb = CLEARPerturbation()
    wPerturb.nncomp_idx = 1
    missing_log_df = pd.DataFrame(columns=['observation', 'feature', 'reason', 'perturbation'])
    i=0
    s1 = pd.Series(results_df.local_data[i], explainer.feature_list)
    s2 = pd.DataFrame(columns=explainer.feature_list)
    s2 = s2.append(s1, ignore_index=True)
    x = symbols('x')
    if  results_df.loc[i, 'features'][0] == '1':
        results_df.loc[i, 'features'].remove('1')
        temp=results_df.loc[i, 'weights'].tolist()
        temp.pop(0)
        results_df.loc[i, 'weights'] = np.array(temp)
    wPerturb.raw_eqn = results_df.loc[i, 'features'].copy()
    wPerturb.raw_weights = results_df.loc[i, 'weights']
    wPerturb.raw_data = results_df.loc[i, 'local_data'].tolist()
    wPerturb.intercept = results_df.loc[i, 'intercept']
    results_df['sufficient']= [getSufficientCauses(wPerturb)]
    counterfactuals_processed = 0
    for counterf_index in range(0, explainer.counterf_rows_df.shape[0]):
        counterfactuals_processed += 1
        wPerturb.target_feature_weight = 0
        wPerturb.target_feature = explainer.counterf_rows_df.loc[counterf_index,'feature']
        # set target probability for b-perturbation
        wPerturb.target_prob = CLEAR_settings.binary_decision_boundary
        # establish if all features are in equation
        if not all(s in wPerturb.raw_eqn  for s in wPerturb.target_feature):
            if missing_log_df.empty:
                idx = 0
            else:
                idx = missing_log_df.index.max() + 1
            missing_log_df.loc[idx, 'observation'] = i
            missing_log_df.loc[idx, 'feature'] = wPerturb.target_feature
            missing_log_df.loc[idx, 'reason'] = 'not in raw equation'
            continue

        # Create equation string
        obsData_df = pd.DataFrame(columns=explainer.feature_list)
        obsData_df.loc[0] = results_df.loc[i, 'local_data']
        str_eqn, wPerturb.target_feature_weight = generateString(explainer, results_df, i, wPerturb)
        str_eqn = str_eqn.replace('x', '0')
        wPerturb.wTx = simplify(str_eqn)
        nncomp_df = catUpdateNncomp_df(explainer, nncomp_df, wPerturb, counterf_index, results_df)

    nncomp_df.observation = nncomp_df.observation.astype(int)
    nncomp_df.reset_index(inplace=True, drop=True)

    """
    Determines the actual values of the AI decision boundary for numeric features. This will then be used 
    for determining the fidelity errors of the CLEAR perturbations.
    """
    return nncomp_df, missing_log_df


def catUpdateNncomp_df(explainer ,nncomp_df, wPerturb, counterf_index, results_df):
    AiProbWithActPerturbation = explainer.counterf_rows_df.loc[counterf_index, 'prediction']
    wPerturb.nncomp_idx += 1
    nncomp_df.loc[wPerturb.nncomp_idx, 'observation'] = 0
    nncomp_df.loc[wPerturb.nncomp_idx, 'feature'] = wPerturb.target_feature
    nncomp_df.loc[wPerturb.nncomp_idx, 'AiProbWithActPerturbation'] = np.float64(AiProbWithActPerturbation)
    nncomp_df.loc[wPerturb.nncomp_idx, 'orgAiProb'] = results_df.loc[0, 'nn_forecast']
    nncomp_df.loc[wPerturb.nncomp_idx, 'orgClass'] = results_df.loc[
        0, 'regression_class']  # needs correcting not sure if regression class needs reporting
    nncomp_df.loc[wPerturb.nncomp_idx, 'orgFeatValue'] = wPerturb.target_feature
    if explainer.data_type == 'image':
        nncomp_df.loc[wPerturb.nncomp_idx, 'actPerturbedFeatValue'] = 'infilled'
    else:
        nncomp_df.loc[wPerturb.nncomp_idx, 'actPerturbedFeatValue'] = explainer.counterf_rows_df.loc[counterf_index, 'feature']
    if CLEAR_settings.regression_type == 'multiple':
        regProbWithActPerturbation = wPerturb.wTx
    else:
        regProbWithActPerturbation = 1 / (1 + exp(-wPerturb.wTx))
    nncomp_df.loc[wPerturb.nncomp_idx, 'regProbWithActPerturbation'] = np.float64(regProbWithActPerturbation)
    nncomp_df.loc[wPerturb.nncomp_idx, 'errorRegProbActPerturb'] = \
        round(abs(regProbWithActPerturbation - AiProbWithActPerturbation), 2)
    return (nncomp_df)


def generateString(explainer, results_df, observation, wPerturb):
    # For categorical target features str_eqn is used to calculate the c-counterfactuals
    raw_data = wPerturb.raw_data
    str_eqn = '+' + str(results_df.loc[observation, 'intercept'])

    for raw_feature in wPerturb.raw_eqn:
        if raw_feature == '1':
            pass
        elif raw_feature in wPerturb.target_feature:
            str_eqn += "+" + str(wPerturb.raw_weights[wPerturb.raw_eqn.index(raw_feature)]) + "*x"
            wPerturb.target_feature_weight = wPerturb.raw_weights[wPerturb.raw_eqn.index(raw_feature)]
        elif raw_feature in explainer.feature_list:
            new_term = raw_data[explainer.feature_list.index(raw_feature)] * wPerturb.raw_weights[
                wPerturb.raw_eqn.index(raw_feature)]
            str_eqn += "+ " + str(new_term)
        elif raw_feature == str(wPerturb.target_feature) + "_sqrd":
            str_eqn += "+" + str(wPerturb.raw_weights[wPerturb.raw_eqn.index(raw_feature)]) + "*x**2"
        elif raw_feature.endswith('_sqrd'):
            new_term = raw_feature.replace('_sqrd', '')
            new_term = (raw_data[explainer.feature_list.index(new_term)] ** 2) * wPerturb.raw_weights[
                wPerturb.raw_eqn.index(raw_feature)]
            str_eqn += "+ " + str(new_term)
        elif '_' in raw_feature:
            interaction_terms = raw_feature.split('_')
            if interaction_terms[0] in wPerturb.target_feature:
                new_term = str(raw_data[explainer.feature_list.index(interaction_terms[1])] \
                               * wPerturb.raw_weights[wPerturb.raw_eqn.index(raw_feature)]) + "*x"
            elif interaction_terms[1] in wPerturb.target_feature:
                new_term = str(raw_data[explainer.feature_list.index(interaction_terms[0])] \
                               * wPerturb.raw_weights[wPerturb.raw_eqn.index(raw_feature)]) + "*x"
            else:
                new_term = str(raw_data[explainer.feature_list.index(interaction_terms[0])]
                               * raw_data[explainer.feature_list.index(interaction_terms[1])]
                               * wPerturb.raw_weights[wPerturb.raw_eqn.index(raw_feature)])
            str_eqn += "+ " + new_term
        else:
            print("error in processing equation string")
        pass
    return str_eqn, wPerturb.target_feature_weight


def Summary_stats(nncomp_df, missing_log_df):
    """ Create summary statistics and frequency histogram
    """
    if nncomp_df.empty:
        print('no data for plot')
        return
    less_target_sd = 0
    temp_df = nncomp_df.copy(deep=True)
    temp_df = temp_df[~temp_df.errorPerturbation.isna()]
    if temp_df['errorPerturbation'].count() != 0:
        less_target_sd = temp_df[temp_df.errorPerturbation <= 0.25].errorPerturbation.count()
        x = temp_df['errorPerturbation']
        x = x[~x.isna()]
        ax = x.plot.hist(grid=True, bins=20, rwidth=0.9)
        plt.title(
            'perturbations = ' + str(temp_df['errorPerturbation'].count()) + '  Freq Counts <= 0.25 sd = ' + str(
                less_target_sd)
            + '\n' + 'regression = ' + CLEAR_settings.regression_type + ', score = ' + CLEAR_settings.score_type
            + ', sample = ' + str(CLEAR_settings.num_samples)
            + '\n' + 'max_predictors = ' + str(CLEAR_settings.max_predictors)
            + ', regression_sample_size = ' + str(CLEAR_settings.regression_sample_size))
        plt.xlabel('Standard Deviations')
        fig = ax.get_figure()
        fig.savefig(CLEAR_settings.CLEAR_path + 'hist' + datetime.now().strftime("%Y%m%d-%H%M") + '.png',
                    bbox_inches="tight")
    else:
        print('no numeric feature data for histogram')
        temp_df = nncomp_df.copy(deep=True)
    # x=np.array(nncomp_df['errorPerturbation'])

    filename1 = CLEAR_settings.CLEAR_path + 'wPerturb_' + datetime.now().strftime("%Y%m%d-%H%M") + '.csv'
    nncomp_df.to_csv(filename1)
    filename2 = CLEAR_settings.CLEAR_path + 'missing_' + datetime.now().strftime("%Y%m%d-%H%M") + '.csv'
    missing_log_df.to_csv(filename2)
    output = [CLEAR_settings.sample_model, less_target_sd]
    filename3 = 'batch.csv'
    try:
        with open(CLEAR_settings.CLEAR_path + filename3, 'a') as file1:
            writes = csv.writer(file1, delimiter=',', skipinitialspace=True)
            writes.writerow(output)
        file1.close()
    except:
        pass
    return


def Single_prediction_report(results_df, nncomp_df, single_regress, explainer):
    if nncomp_df.empty:
        print('no counterfactuals found')
    if len(explainer.class_labels)==2:
        explanandum= explainer.class_labels[1]



    def round_sig(x, sig=2):
        if type(x) == np.ndarray:
            x = x[0]
        if x == 0:
            y= 0
        else:
            y= round(x, sig - int(floor(log10(abs(x)))) - 1)
        return y

    j = results_df.index.values[0]
    if CLEAR_settings.regression_type == 'multiple':
        regression_formula = 'prediction = ' + str(round_sig(results_df.intercept[j]))
    else:
        regression_formula = '<font size = "4.5">prediction =  [ 1 + e<sup><b>-w<sup>T</sup>x</sup></b> ]<sup> -1</sup></font size><br><br>' \
                             + '<font size = "4.5"><b><i>w</i></b><sup>T</sup><b><i>x</font size></i></b> =  ' + str(
            round_sig(results_df.intercept[j]))

    for i in range(len(results_df.features[j])):
        if ("_" in results_df.features[j][i]) and ("_sqrd" not in results_df.features[j][i]):
            results_df.features[j][i] = "(" + results_df.features[j][i] + ")"
        for h in range(results_df.features[j][i].count("Dd")):
            t = results_df.features[j][i].find("Dd")
            if t == 3 and len(results_df.features[j][i]) > 6:
                results_df.features[j][i] = results_df.features[j][i][5:]
            elif t == 3 and len(results_df.features[j][i]) <= 6:
                results_df.features[j][i] = results_df.features[j][i][0:3] + results_df.features[j][i][t + 2:]
            elif len(results_df.features[j][i][t + 2:]) > 2:
                results_df.features[j][i] = results_df.features[j][i][:t - 3] + results_df.features[j][i][t + 2:]
            else:
                results_df.features[j][i] = results_df.features[j][i][:t] + results_df.features[j][i][t + 2:]
        if results_df.features[j][i] == '1':
            continue
        elif results_df.weights[j][i] < 0:
            regression_formula = regression_formula + ' - ' + str(-1 * round_sig(results_df.weights[j][i])) + \
                                 ' ' + results_df.features[j][i]
        else:
            regression_formula = regression_formula + ' + ' + str(round_sig(results_df.weights[j][i])) + \
                                 ' ' + results_df.features[j][i]
    regression_formula = regression_formula.replace("_sqrd", "<sup>2</sup>")
    regression_formula = regression_formula.replace("_", "*")
    report_AI_prediction = str(round_sig(results_df.nn_forecast[j]))
    if CLEAR_settings.score_type == 'adjR':
        regression_score_type = "Adjusted R-Squared"
    else:
        regression_score_type = CLEAR_settings.score_type

    # get rid of dummy variables equal to zero
    temp2_df = pd.DataFrame(columns=['Feature', 'Input Value'])
    temp = [col for col in single_regress.data_row.columns \
            if not ((single_regress.data_row.loc[0, col] == 0) and (col in explainer.cat_features))]
    input_data = single_regress.data_row.loc[0, temp]
    k = 0
    for col in input_data.index:
        if col in explainer.cat_features:
            if explainer.data_type == 'image':
                temp2_df.loc[k, 'Feature']=col[5:]
            else:
                temp2_df.loc[k, 'Feature'] = col.replace("Dd", "=")
            temp2_df.loc[k, 'Input Value'] = "1"
        else:
            temp2_df.loc[k, 'Feature'] = col
            temp2_df.loc[k, 'Input Value'] = str(round(input_data.iloc[k], 2))
        k += 1
    inputData_df = temp2_df.copy(deep=True)
    inputData_df.set_index('Feature', inplace=True)
    inputData_df = inputData_df.transpose().copy(deep=True)

    #create counterfactual tables
    temp_df = nncomp_df.copy(deep=True)
    for index, rows in temp_df.iterrows():
        features = [[x[-5:] for x in temp_df.loc[index,'feature']]]
        temp_df.at[index,'feature'] = features
        temp_df.loc[index,'actPerturbedFeatValue'] = 'infilled'

    c_counter_df = temp_df[['feature', 'actPerturbedFeatValue', 'AiProbWithActPerturbation',
                            'regProbWithActPerturbation', 'errorRegProbActPerturb']].copy()
    c_counter_df.rename(columns={"actPerturbedFeatValue": "value",
                                 "AiProbWithActPerturbation": "AI using c-counterfactual value",
                                 "regProbWithActPerturbation": "regression forecast using c-counterfactual",
                                 "errorRegProbActPerturb": "regression forecast error"},
                        inplace=True)

    # sorted unique feature list for the 'select features' checkbox
    feature_box = results_df.features[j]
    feature_box = ",".join(feature_box).replace('(', '').replace(')', '').replace('_', ',').split(",")
    feature_box = sorted(list(set(feature_box)), key=str.lower)
    for x in ['sqrd', '1']:
        if x in feature_box:
            feature_box.remove(x)
    # results_df.weights needs pre-processing prior to sending to HTML
    weights = results_df.weights.values[0]
    spreadsheet_data = results_df.spreadsheet_data.values[0]
    if len(weights) == len(spreadsheet_data) + 1:
        weights = np.delete(weights, [0])
    weights = weights.tolist()

    # calculate feature importance
    feat_importance_df = pd.DataFrame(columns=feature_box)
    for y in feature_box:
        temp = 0
        cnt = 0
        for z in results_df.features[j]:
            if y in z:
                if y == z:
                    temp += results_df.weights[j][cnt] * results_df.spreadsheet_data[j][cnt]
                elif '_sqrd' in z:
                    temp += results_df.weights[j][cnt] * (results_df.spreadsheet_data[j][cnt])
                else:
                    temp += (results_df.weights[j][cnt] * results_df.spreadsheet_data[j][cnt]) / 2
            cnt += 1
        feat_importance_df.loc[0, y] = temp
    # normalise by largest absolute value
    t = feat_importance_df.iloc[0, :].abs()
    top_seg = pd.to_numeric(t).idxmax()
    # create feature importance bar chart
    max_display = min(feat_importance_df.shape[1],10) #select top 10 features
    counterfactual_segs = []  #add any missing counterfactuuals
    for index, rows in c_counter_df.iterrows():
        for index2 in range(0,len(c_counter_df.loc[index,'feature'])):
            temp = c_counter_df.loc[index, 'feature'][index2]
            if temp not in counterfactual_segs:
                counterfactual_segs += temp
    counterfactual_segs= list(set(counterfactual_segs))
    for index, rows in c_counter_df.iterrows():
        feature = c_counter_df.at[index, 'feature']
        feature = str(feature).replace('[', "")
        feature = str(feature).replace(']', "")
        c_counter_df.at[index, 'feature'] = feature
    #counterfactual_segs = [np.int32(i[-2:]) for i in counterfactual_segs]
    #so need index of counterfactual_segs in feature_importance_df
    temp = feat_importance_df.columns.to_list()
    counterfactual_idx= [temp.index(i) for i in counterfactual_segs]
    counterfactual_idx = np.array(counterfactual_idx)
    display_features = np.sort(feat_importance_df.abs().values.argsort(1)[0][-max_display:])
    display_features = np.union1d(display_features,counterfactual_idx)
    ax = feat_importance_df.iloc[:, display_features].plot.barh(width=1)
    ax.patches[0].set_color([0.08,0.05,1])  # sets colour for 'background' segment
    ax.legend(fontsize=12)
    ax.invert_yaxis()
    ax.margins(y=0)
    ax.yaxis.set_visible(False)
    leg = ax.legend()
    patches = leg.get_patches()
    bar_colours = [x.get_facecolor() for x in patches]
    fig = ax.get_figure()
    fig.set_figheight(5)
    fig.set_figheight(6.5)
    fig.tight_layout()
    fig.savefig('Feature_plot.png', bbox_inches="tight")

    #Create regression scatterplot
    fig = plt.figure()
    plt.scatter(single_regress.neighbour_df.loc[:, 'prediction'], single_regress.after_center_option, c='green',
                s=10)
    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), c="red", linestyle='-')
    plt.xlabel('Target AI System')
    if CLEAR_settings.regression_type == 'logistic':
        plt.ylabel('CLEAR Logistics Regression')
    elif CLEAR_settings.regression_type == 'multiple':
        plt.ylabel('CLEAR Multiple Regression')
    else:
        plt.ylabel('CLEAR Polynomial Regression')
    fig.savefig('CLEAR_plot.png', bbox_inches="tight")
    pd.set_option('colheader_justify', 'left', 'precision', 2)
    env = Environment(loader=FileSystemLoader('.'))

    #create segmented image
    if explainer.data_type == 'image':
        positive_features = np.where(feat_importance_df.values > 0)[1]
        positive_features = positive_features[np.in1d(positive_features, display_features)]
        negative_features = np.where(feat_importance_df.values < 0)[1]
        negative_features = negative_features[np.in1d(negative_features, display_features)]
        # save_img('org_image.png',explainer.image[0])
        feature_num = [np.int64(x[3:]) for x in feature_box]
        for k in [positive_features, negative_features]:
            colour_seg = np.ones(explainer.image[0].shape)
            for m in k:
                bar_idx = np.where(m == display_features)[0][0]
                for n in range(3):
                    colour_seg[:, :, n][explainer.segments == feature_num[m]] = bar_colours[bar_idx][n]
                bar_idx += 1
            im1=mark_boundaries(explainer.image[0], explainer.segments).astype(np.float64)
            im1 =(im1- im1.min())/(im1.max()-im1.min()) #normalises between 0 and 1
            im2=cv2.addWeighted(im1, 0.5, colour_seg, 0.5, 0)
            if CLEAR_settings.debug_mode is True:
                plt.imshow(im2)
                plt.show()
            if k in positive_features:
                plt.imsave('pos_segmented.png', im2, dpi=3000)
            else:
                plt.imsave('neg_segmented.png', im2, dpi=3000)

    # create heatmap
        heat_array = np.zeros(explainer.image[0,:,:,0].shape)
        for m, n in enumerate(feature_num):
            heat_array[explainer.segments == n]= feat_importance_df.iloc[0,m]
        unique, counts = np.unique(heat_array, return_counts=True)
        for cnt, k in enumerate(unique):
            heat_array[heat_array == k] = k /counts[cnt]
        heat_array[np.absolute(heat_array) < heat_array.max()/100] = 0
        pointing_game = True
        if pointing_game is True:
            # pointing_array = np.absolute(heat_array)
            np.savetxt(CLEAR_settings.CLEAR_path + 'CLEAR_VGG_point_heatmap_' + explainer.batch +
                         '.csv', heat_array, delimiter=',')
        offset = mcolors.TwoSlopeNorm(vcenter=0)
        if CLEAR_settings.debug_mode is True:
            plt.imshow(plt.cm.seismic(offset(heat_array))) #was cm.bwr
            plt.show()
        buf = io.BytesIO()
        plt.imsave(buf, plt.cm.seismic(offset(heat_array)))
        buf.seek(0)
        heatmap= Image.open(buf).convert("RGB")
        heatmap = np.array(heatmap)
        heatmap= (heatmap/255).astype(np.float32)
        heatmap = cv2.addWeighted(explainer.image[0], 0.1, heatmap, 0.9, 0)
        if CLEAR_settings.debug_mode is True:
            plt.imshow(heatmap)
            plt.show()
        if explainer.batch == 'None':
            plt.imsave('CLEAR_heatmap.png', heatmap, dpi=3000)
        else:
            plt.imsave( CLEAR_settings.CLEAR_path+ 'CLEAR_heatmap_' + explainer.batch +
                         '.png', heatmap, dpi=3000)
    #write to HTML
        if  len(results_df.loc[0,'sufficient'])==0:
            sufficient_causes_out = []
        else:
            sufficient_causes_out = [x[-5:] for x in results_df.loc[0,'sufficient']]
        template = env.get_template("CLEAR_Image_report.html")
        template_vars = {"title": "CLEAR Statistics",
                         "input_data_table": inputData_df.to_html(index=False, classes='mystyle'),
                         "dataset_name": CLEAR_settings.case_study,
                         "explanadum": explanandum,
                         "observation_number": j,
                         "regression_formula": regression_formula,
                         "prediction_score": round_sig(results_df.Reg_Score[j]),
                         "regression_score_type": regression_score_type,
                         "regression_type": CLEAR_settings.regression_type,
                         "AI_prediction": report_AI_prediction,
                         "cat_counterfactual_table": c_counter_df.to_html(index=False, classes='mystyle'),
                         "feature_list": feature_box,
                         "spreadsheet_data": spreadsheet_data,
                         "weights": weights,
                         "intercept": results_df.intercept.values[0],
                         "sufficient":  sufficient_causes_out
                         }
        with open('CLEAR_Image_report_full.html', 'w') as fh:
            fh.write(template.render(template_vars))
    batch_Xrays = True
    if batch_Xrays is True:
        if c_counter_df.empty is True:
            c_counter_df.loc[0,'feature']= 'None'
        c_counter_df['threshold'] = CLEAR_settings.image_segment_threshold
        c_counter_df['regression score']=round_sig(results_df.Reg_Score[j])
        c_counter_df['Xray_ID'] = explainer.batch
        c_counter_df['top_seg'] = top_seg
        c_counter_df['Seg00_too_large'] = False
        c_counter_df['poor_data']= explainer.poor_data
        if 'Seg00' in feature_box:
            if feat_importance_df.loc[0,'Seg00']/feat_importance_df.loc[0,top_seg]>0.5:  #was 0.25
               c_counter_df['Seg00_too_large'] = True
        # c_counter_df.to_pickle(CLEAR_settings.CLEAR_path+'c_counter_df.pkl')

        plt.close('all')
    return(c_counter_df)

