import pandas as pd
import numpy as np
import re

from gbychain.code.groupby_result import GroupbyResult


class GroupbyChainOperator:
    def __init__(self, init_df, init_name="init"):
        if init_df is None:
            raise ValueError("Please set init_df")

        self.init_df = init_df
        self.init_name = init_name
        self.level = 0
        self.level_result_dict = {str(self.level): GroupbyResult(self.init_name, "None",
                                                                 None, None, None,
                                                                 None, None, None,
                                                                 init_df, None, self.level)}
        self.level_column_dict = {}

    def check_update_inputs(self, tgt_names=None, func_dict=None):
        prev_level = self.level
        input_result = self.level_result_dict[str(prev_level)]
        input_df = input_result.output_df
        input_name = input_result.output_name

        if tgt_names is None:
            tgt_names = input_result.tgt_names
        if func_dict is None:
            func_dict = input_result.func_dict
        if tgt_names is None or func_dict is None:
            raise ValueError("Please set tgt_names and func_dict as they are not included in the previous result.")

        return input_df, input_name, tgt_names, func_dict

    def update_level_result(self, new_df, operation_type, input_name, groups, indices, tgt_names, func_dict, funcs_name,
                            columns_dict, output_name):
        level = self.level + 1
        result = GroupbyResult(output_name, operation_type, input_name, groups, indices, tgt_names, func_dict,
                               funcs_name, new_df, columns_dict, level)
        self.level_column_dict[str(level)] = columns_dict
        self.level_result_dict[str(level)] = result
        self.level = level

    def update_by_agg(self, groups, indeces=None, tgt_names=None, func_dict=None, funcs_name=None, output_name=None,
                      is_debug=False):
        input_df, input_name, tgt_names, func_dict = self.check_update_inputs(tgt_names, func_dict)

        agg_df, columns_dict = agg_by_groups(input_df, groups, tgt_names, func_dict, True, is_debug=is_debug)
        if is_debug:
            print(tgt_names)

        gp_inx_names = groups + indeces if indeces is not None else groups
        tgt_names2 = list(columns_dict.keys())
        self.update_level_result(agg_df, "agg", input_name, groups, indeces, tgt_names2, func_dict,
                                 funcs_name, columns_dict, output_name)
        return self

    def update_by_apply2_agg(self, groups, indices=None, tgt_names_func_list=None, output_name=None):
        prev_level = self.level
        input_result = self.level_result_dict[str(prev_level)]
        input_df = input_result.output_df
        input_name = input_result.output_name

        # tgt_names = [s[0] for s in tgt_names_func_list]
        # funcs = [s[2] for s in tgt_names_func_list]
        apply_df, columns_dict = apply_agg_by_groups(input_df, groups, tgt_names_func_list, True)
        tgt_names2 = apply_df.drop(groups, axis=1).columns
        self.update_level_result(apply_df, "apply2-agg", input_name, groups, indices, tgt_names2,
                                 None, None,
                                 columns_dict, output_name)
        return self

    def update_by_transform(self, groups, indices=None, tgt_names=None, func_dict=None, funcs_name=None,
                            output_name=None, is_debug=False):
        input_df, input_name, tgt_names, func_dict = self.check_update_inputs(tgt_names, func_dict)

        tf_df, columns_dict = tf_by_groups(input_df, groups, tgt_names, func_dict, True)
        tf_df = tf_df.drop(tgt_names, axis=1)
        tgt_names2 = list(columns_dict.keys())
        if is_debug:
            print(tgt_names2)
        self.update_level_result(tf_df, "transform", input_name, groups, indices, tgt_names2, func_dict,
                                 funcs_name, columns_dict, output_name)
        return self

    # ... [他のメソッドや補助関数]

    def get_output_df(self, level, use_func_order=True):
        result = self.level_result_dict[str(level)]
        output_df = result.output_df.copy()
        if use_func_order:
            return output_df
        else:
            if level == 0:
                return output_df
            else:

                for l in np.arange(1, level + 1)[::-1]:
                    tgt_dict = self.level_column_dict[str(l)]
                    pattern = re.compile(r"(_\d){" + str(l) + r"}$")
                    tgt_dict2 = {key: value for key, value in tgt_dict.items() if pattern.search(key)}
                    # print(l)
                    # print(tgt_dict)
                    # print("---------------")
                    # print(output_df.columns)
                    for old, new in tgt_dict2.items():
                        output_df.columns = [col.replace(old, new) for col in output_df.columns]
                    # print(output_df.columns)
                return output_df

    def merge_all_level_df(self, use_func_order=True):
        ret_df = self.get_output_df(0, use_func_order)
        for l in np.arange(self.level):
            l_ret = self.level_result_dict[str(l + 1)]
            method = l_ret.method
            groups = l_ret.groups
            output_df = self.get_output_df(l + 1, use_func_order)
            if method == "tf":
                ret_df = pd.concat(
                    [ret_df, output_df.drop(groups, axis=1)],
                    axis=1)
            else:
                ret_df = ret_df.merge(output_df, on=groups, how="left")
        return ret_df

    def get_log(self):
        return [str(v) for _, v in self.level_result_dict.items()]


def apply_agg_by_groups(df, groups, col2_func_tpl_list, use_func_order=False
                        ):
    def apply_funcs(x):
        results = {}
        for left, right, func in col2_func_tpl_list:
            results[f'{left}-{right}_' + func.__name__] = func(x, left, right)
        return pd.Series(results)

    applied_df = df.groupby(groups).apply(apply_funcs)

    def remove_last_element(s):
        parts = s.split("_")
        return "_".join(parts[:-1])

    # applied_df = restore_column_dtypes(applied_df, dtypes_dict)
    if use_func_order:
        # new_columns = [f"apply2-agg_{str(inx)}" for inx, col in enumerate(applied_df.columns)]
        new_columns = [f"{remove_last_element(col)}_{str(inx)}"
                       for inx, col in enumerate(applied_df.columns)]
        old_columns = applied_df.columns
        applied_df.columns = new_columns
        columns_map = dict(zip(new_columns, old_columns))
        return applied_df.reset_index(), columns_map

    return applied_df.reset_index()


def agg_by_groups(df, groups, tgt_names, func_dict, use_func_order=False, is_debug=False):
    # グループごとに関数を適用するための辞書を作成
    agg_funcs = {col: list(func_dict.values()) for col in tgt_names}
    if is_debug:
        print(agg_funcs)
    # グループごとに関数を適用
    agg_df = df.groupby(groups)[tgt_names].agg(agg_funcs)
    # 列名のフラット化
    # agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    agg_df.columns = [f'{col}_{fname}' for col in tgt_names for fname in func_dict.keys()]
    if use_func_order:
        new_columns = [f'{col}_{idx}' for col in tgt_names for idx, _ in enumerate(func_dict)]
        old_columns = agg_df.columns
        agg_df.columns = new_columns
        columns_map = dict(zip(new_columns, old_columns))
        return agg_df.reset_index(), columns_map

    # 結果を表示
    return agg_df.reset_index()


def tf_by_groups(df, groups, tgt_names, func_dict, use_func_order=False, is_debug=False):
    tf_funcs = {col: func_dict for col in tgt_names}
    df_copy = df.loc[:, groups + tgt_names].copy()
    del df
    df_copy.loc[:, tgt_names] = df_copy.loc[:, tgt_names].astype(np.float64)
    old_columns = []
    new_columns = []
    for col, func_dict_ in tf_funcs.items():
        no = 0
        for fname, fvalue in func_dict_.items():
            df_copy.loc[:, col + "_" + fname] = df_copy.groupby(groups)[col].transform(fvalue)
            old_columns.append(col + "_" + fname)
            new_columns.append(col + "_" + str(no))
            no = no + 1

    if use_func_order:
        df_copy = df_copy.rename(columns=dict(zip(old_columns, new_columns)))
        columns_map = dict(zip(new_columns, old_columns))
        return df_copy, columns_map
