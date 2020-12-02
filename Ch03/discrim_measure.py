import warnings
warnings.filterwarnings("ignore") ## indexing issue that does not affect these computations

## simple but realistic and legally relevant example: simple ratios of success rates
import pandas as pd
import numpy as np

## p 92
data = pd.read_csv("german.data", delim_whitespace = True, header = None)

success_data = data.groupby(8).agg([lambda x: sum(x == 1),
                                    lambda x: len(x)]).iloc[:, -2:]
success_data.columns = ["num_success", "num_total"]
most_successful = np.max(success_data.num_success / success_data.num_total)
print(success_data.num_success / success_data.num_total / most_successful)

## p 94
## more nuanced example
## looking at whether a rule is alpha discriminatory
## note code is updated as compared to book to reflect code deprecations
cond1                                    = data.iloc[:, 8]  == "A91"
num_div_males                            = data[cond1].shape[0]

cond2                                    = data.iloc[:, 5]  == "A65"
num_unknown_credit                       = data[cond2].shape[0]

cond3                                    = data.iloc[:, 5]  == "A65"
cond4                                    = data.iloc[:, 8]  == "A91"
num_unknown_credit_div_male              = data[cond3][cond4].shape[0]

cond5                                    = data.iloc[:, 20] == 2
num_bad_outcome                          = data[cond5].shape[0]

cond6                                    = data.iloc[:, 5]  == "A65"
cond7                                    = data.iloc[:, 20] == 2
num_bad_outcome_unknown_credit           = data[cond6][cond7].shape[0]

cond8                                    = data.iloc[:, 8]  == "A91"
cond9                                    = data.iloc[:, 5] == "A65"
cond10                                   = data.iloc[:, 20] == 2
num_bad_outcome_unknown_credit_div_male  = data[cond8][cond9][cond10].shape[0]

### test association rule 1: unknown credit -> bad outcome
rule1_conf = num_bad_outcome_unknown_credit / num_unknown_credit
rule2_conf = num_bad_outcome_unknown_credit_div_male / num_unknown_credit_div_male

## compute elift (ratio of confidence rules)
print(rule2_conf / rule1_conf)

## now rerun for divorced females
## extra example
## not in book
num_div_females                         = data[data.iloc[:, 8]  == "A92"].shape[0]
num_unknown_credit                      = data[data.iloc[:, 5]  == "A65"].shape[0]
num_unknown_credit_div_female           = data[data.iloc[:, 5]  == "A65"][data.iloc[:, 8]  == "A92"].shape[0]
num_bad_outcome                         = data[data.iloc[:, 20] == 2].shape[0]
num_bad_outcome_unknown_credit          = data[data.iloc[:, 5]  == "A65"][data.iloc[:, 20] == 2].shape[0]
num_bad_outcome_unknown_credit_div_female = data[data.iloc[:, 8]  == "A92"][data.iloc[:, 5] == "A65"][data.iloc[:, 20] == 2].shape[0]

### test association rule 1: unknown credit -> bad outcome
rule1_conf = num_bad_outcome_unknown_credit / num_unknown_credit
rule2_conf = num_bad_outcome_unknown_credit_div_female / num_unknown_credit_div_female

## compute elift (ratio of confidence rules)
print(rule2_conf / rule1_conf)
