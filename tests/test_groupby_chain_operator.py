from unittest import TestCase, main
import pandas as pd
import numpy as np

from gbychain.code.groupby_chain_operator import GroupbyChainOperator


def make_input_df():
    df = pd.DataFrame(
        {
            "group1": ["A"] * 6 + ["B"] * 4,
            "group2": ["a"] * 3 + ["b"] * 4 + ["c"] * 3,
            "v1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 100],
            "v2": np.arange(20, 30)[::-1]
        }
    )
    return df


def gen_gcop_update_by_agg(df):
    func_dict = {"max": "max", "min": "min"}
    func_dict2 = {"sum": "sum", "zero": lambda x: 0.0}
    gcop = GroupbyChainOperator(df)
    return (
        gcop.update_by_agg(["group1", "group2"], tgt_names=["v1", "v2"], func_dict=func_dict)
        .update_by_agg(["group1"], func_dict=func_dict2)
    )


def gen_gcop_update_by_apply2_agg(df):
    def lessthan(group, col1, col2):
        return (group[col1] > group[col2]).sum()

    tgt_names_func_list = [("v1", "v2", lessthan)]

    gcop = GroupbyChainOperator(df)
    return (
        gcop.update_by_apply2_agg(["group1", "group2"], "", tgt_names_func_list)
        .update_by_agg(["group1", "group2"], func_dict={"sum": "sum"})

    )


def gen_gcop_update_by_transform(df):
    def ewm_array(arr, span):
        n = len(arr)
        ewma = np.empty(n, dtype=np.float64)
        alpha = 2 / (span + 1)
        ewma[0] = arr[0]
        for i in range(1, n):
            ewma[i] = alpha * arr[i] + (1 - alpha) * ewma[i - 1]
        return ewma

    func_dict = {"cumsum": "cumsum"}
    func_dict2 = {
        "ewm1": lambda x: ewm_array(x.values, 1),
        "ewm3": lambda x: ewm_array(x.values, 3),
    }
    # func_dict2 = {"sum":"sum"}
    # df2 = df.drop("v2", axis=1)
    gcop = GroupbyChainOperator(df)
    return (
        gcop.update_by_transform(["group1", "group2"], tgt_names=["v1"], func_dict=func_dict)
        .update_by_transform(["group1", "group2"], func_dict=func_dict2)
    )


def verify_update_dfs(e_level1_df, e_level2_df, e_level1_order_df, e_level2_order_df,df, gcop_agg):
    verify_two_dfs(e_level1_df, gcop_agg.get_output_df(1, False), "Check level=1 result")
    verify_two_dfs(e_level2_df, gcop_agg.get_output_df(2, False), "Check level=2 result")
    verify_two_dfs(e_level1_order_df, gcop_agg.get_output_df(1, True),
                   "Check level=1 result with use_func_order=True")
    verify_two_dfs(e_level2_order_df, gcop_agg.get_output_df(2, True),
                   "Check level=2 result with use_func_order=True")

    e_merge_df = df.merge(e_level1_df).merge(e_level2_df)
    e_merge_order_df = df.merge(e_level1_order_df).merge(e_level2_order_df)
    verify_two_dfs(e_merge_df, gcop_agg.merge_all_level_df(False), "Check merge_all_level_df")
    verify_two_dfs(e_merge_order_df, gcop_agg.merge_all_level_df(True),
                   "Check merge_all_level_df with use_func_order=True")


def verify_two_dfs(e_df, a_df, title=None):
    if title is not None:
        print(f"--------{title}-----------")
    print("--------Expected-----------")
    print(e_df)
    print("--------Actual-----------")
    print(a_df)
    try:
        pd.testing.assert_frame_equal(e_df, a_df, check_dtype=False)
        print("\n")
    except AssertionError as e:
        raise AssertionError("DataFrame comparison failed.")


def verify_two_lists(e_list, a_list):
    print("--------Expected List-----------")
    print(e_list)
    print("--------Actual List-----------")
    print(a_list)

    # リストの長さが同じかどうかを確認
    assert len(e_list) == len(a_list), "Lists have different lengths."

    # 各要素を比較
    for e_item, a_item in zip(e_list, a_list):
        # 数値の場合、型を無視して値のみを比較
        if isinstance(e_item, (int, float)) and isinstance(a_item, (int, float)):
            assert e_item == a_item, f"Items do not match: {e_item} != {a_item}"
        else:
            # 数値以外の場合、正確に一致することを確認
            assert e_item == a_item, f"Items do not match: {e_item} != {a_item}"

    print("Lists are equal.")


class GroupbyChainOperatorTest(TestCase):

    def setUp(self):
        self.df = make_input_df()
        self.gcop_agg = gen_gcop_update_by_agg(self.df)
        self.gcop_apply2_agg = gen_gcop_update_by_apply2_agg(self.df)
        print(self.df)
        self.gcop_tf = gen_gcop_update_by_transform(self.df)


    def test_update_by_agg(self):
        pd.set_option('display.max_columns', 50)
        e_level1_df = pd.DataFrame([
            {'group1': 'A', 'group2': 'a', 'v1_max': 2, 'v1_min': 0, 'v2_max': 29, 'v2_min': 27},
            {'group1': 'A', 'group2': 'b', 'v1_max': 5, 'v1_min': 3, 'v2_max': 26, 'v2_min': 24},
            {'group1': 'B', 'group2': 'b', 'v1_max': 6, 'v1_min': 6, 'v2_max': 23, 'v2_min': 23},
            {'group1': 'B', 'group2': 'c', 'v1_max': 100, 'v1_min': 7, 'v2_max': 22, 'v2_min': 20},
        ])
        e_level2_df = pd.DataFrame([
            {'group1': 'A',
             'v1_max_sum': 7, 'v1_max_zero': 0.0, 'v1_min_sum': 3, 'v1_min_zero': 0.0,
             'v2_max_sum': 29 + 26, 'v2_max_zero': 0.0, 'v2_min_sum': 27 + 24, 'v2_min_zero': 0.0,
             },
            {'group1': 'B',
             'v1_max_sum': 106, 'v1_max_zero': 0.0, 'v1_min_sum': 13, 'v1_min_zero': 0.0,
             'v2_max_sum': 23 + 22, 'v2_max_zero': 0.0, 'v2_min_sum': 23 + 20, 'v2_min_zero': 0.0,
             },

        ])
        e_level1_order_df = pd.DataFrame([
            {'group1': 'A', 'group2': 'a', 'v1_0': 2, 'v1_1': 0, 'v2_0': 29, 'v2_1': 27},
            {'group1': 'A', 'group2': 'b', 'v1_0': 5, 'v1_1': 3, 'v2_0': 26, 'v2_1': 24},
            {'group1': 'B', 'group2': 'b', 'v1_0': 6, 'v1_1': 6, 'v2_0': 23, 'v2_1': 23},
            {'group1': 'B', 'group2': 'c', 'v1_0': 100, 'v1_1': 7, 'v2_0': 22, 'v2_1': 20},
        ])
        e_level2_order_df = pd.DataFrame([
            {'group1': 'A',
             'v1_0_0': 7, 'v1_0_1': 0.0, 'v1_1_0': 3, 'v1_1_1': 0.0,
             'v2_0_0': 29 + 26, 'v2_0_1': 0.0, 'v2_1_0': 27 + 24, 'v2_1_1': 0.0,
             },
            {'group1': 'B',
             'v1_0_0': 106, 'v1_0_1': 0.0, 'v1_1_0': 13, 'v1_1_1': 0.0,
             'v2_0_0': 23 + 22, 'v2_0_1': 0.0, 'v2_1_0': 23 + 20, 'v2_1_1': 0.0,
             },

        ])
        verify_update_dfs(e_level1_df, e_level2_df, e_level1_order_df, e_level2_order_df, self.df, self.gcop_agg)

    def test_update_by_apply2_agg(self):
        pd.set_option('display.max_columns', 50)
        e_level1_df = pd.DataFrame([
            {'group1': 'A', 'group2': 'a', 'v1-v2_lessthan': 0},
            {'group1': 'A', 'group2': 'b', 'v1-v2_lessthan': 0},
            {'group1': 'B', 'group2': 'b', 'v1-v2_lessthan': 0},
            {'group1': 'B', 'group2': 'c', 'v1-v2_lessthan': 1}
        ])
        e_level2_df = pd.DataFrame([
            {'group1': 'A', 'group2': 'a', 'v1-v2_lessthan_sum': 0},
            {'group1': 'A', 'group2': 'b', 'v1-v2_lessthan_sum': 0},
            {'group1': 'B', 'group2': 'b', 'v1-v2_lessthan_sum': 0},
            {'group1': 'B', 'group2': 'c', 'v1-v2_lessthan_sum': 1}
        ])
        e_level1_order_df = pd.DataFrame([
            {'group1': 'A', 'group2': 'a', 'v1-v2_0': 0},
            {'group1': 'A', 'group2': 'b', 'v1-v2_0': 0},
            {'group1': 'B', 'group2': 'b', 'v1-v2_0': 0},
            {'group1': 'B', 'group2': 'c', 'v1-v2_0': 1}
        ])
        e_level2_order_df = pd.DataFrame([
            {'group1': 'A', 'group2': 'a', 'v1-v2_0_0': 0},
            {'group1': 'A', 'group2': 'b', 'v1-v2_0_0': 0},
            {'group1': 'B', 'group2': 'b', 'v1-v2_0_0': 0},
            {'group1': 'B', 'group2': 'c', 'v1-v2_0_0': 1}

        ])

        verify_update_dfs(e_level1_df, e_level2_df, e_level1_order_df, e_level2_order_df, self.df, self.gcop_apply2_agg)


    def test_update_by_tf(self):
        pd.set_option('display.max_columns', 50)
        e_level1_df = pd.DataFrame(
            {
                "group1": ["A"] * 6 + ["B"] * 4,
                "group2": ["a"] * 3 + ["b"] * 4 + ["c"] * 3,
                "v1_cumsum": [0,1,3] + [3,7,12] + [6] + [7, 15, 115]
                # "v2_cumsum": [29,57,84] + [26,51,75] + [23] + [22, 43, 63],
            }
        )
        e_level2_df = pd.DataFrame(
            {
                "group1": ["A"] * 6 + ["B"] * 4,
                "group2": ["a"] * 3 + ["b"] * 4 + ["c"] * 3,
                "v1_cumsum_ewm1": [0,1,3] + [3,7,12] + [6] + [7, 15, 115],
                "v1_cumsum_ewm3": [0, 0 * 0.5 + 1.0 * 0.5, 0.5 * 0.5 + 3.0 * 0.5 ] +
                                  [3, 3.0 * 0.5 + 7.0 * 0.5, 5.0 * 0.5 + 12 * 0.5] +
                                  [6] +
                                  [7.0, 7.0 * 0.5 + 15.0 * 0.5, 11.0 * 0.5 + 115 * 0.5]

            }
        )
        e_level1_order_df = e_level1_df.rename(columns={"v1_cumsum":"v1_0"})
        e_level2_order_df = e_level2_df.rename(columns={"v1_cumsum_ewm1":"v1_0_0", "v1_cumsum_ewm3":"v1_0_1"})
        verify_update_dfs(e_level1_df, e_level2_df, e_level1_order_df, e_level2_order_df, self.df, self.gcop_tf)

if __name__ == '__main__':
    main()
