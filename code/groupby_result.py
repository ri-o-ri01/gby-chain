import pandas as pd

def make_groupby_names(groups):
    return "-".join(["".join([part[0] for part in e.split("_")]) for e in groups])

class GroupbyResult:
    def __init__(self, output_name: str, method: str, input_name: str, groups: list, indices: str, tgt_names: list,
                 func_dict: dict, funcs_name: str, output_df: pd.DataFrame, output_column_dict: dict, level: int):

        # if output_df is None:
        #     self.output_name = ""
        # else:
        self.output_name = output_name
        self.method = method
        self.input_name = input_name
        self.groups = groups
        self.indices = indices
        self.tgt_names = tgt_names
        self.func_dict = func_dict
        self.funcs_name = funcs_name
        self.output_df = output_df
        self.output_column_dict = output_column_dict
        self.level = level


    # def __str__(self):
    def __str__(self):
        if self.level == 0:
            return self.output_name + "_" + str(self.level)
        else:
            if self.output_name is None:
                groupby_names = make_groupby_names(self.groups)
                values = [
                    # "GroupbyResult",
                    self.method, groupby_names, self.funcs_name,
                    # self.input_name, self.output_name,
                    self.level
                ]
                return "_".join([str(v) for v in values if v is not None])
            else:
                return self.output_name + "_" + str(self.level)