import numpy as np
import pandas as pd
from pandas import DataFrame


class FeatureDictManage:
    """
    主要作用：将DataFrame中的离散字段映射成递增的数字，
    主要包含两步：
        1. 构建映射关系：构建离散字段->ID的映射
        2. 字段映射：将字段映射到相应的ID
    """

    def __init__(self):
        self.feature_size = {}
        self.feature_map = {}

    @staticmethod
    def _field_converse(dataframe: DataFrame, fields=None) -> dict:
        if fields is None:
            fields = dict(zip(dataframe.columns, dataframe.columns))
        elif isinstance(fields, str):
            fields = {fields: fields}
        elif isinstance(fields, list):
            fields = dict(zip(fields, fields))
        return fields

    def add_feature(self, dataframe: DataFrame, fields=None):
        """
        构建映射关系
        :param dataframe: 数据
        :param fields: 需要构建映射关系的字段，可以是str,list,map  {data_field : save_field}
        """
        fields = self._field_converse(dataframe, fields)

        for field_k, field_v in fields.items():
            if field_v in self.feature_size.keys():
                field_list = set(dataframe[field_k].drop_duplicates(
                ).values) - set(self.feature_map.get(field_v, []))
                field_list = list(field_list)
                field_list.sort()
                d = dict(zip(field_list, [
                         i + self.feature_size.get(field_v, 0) for i in range(len(field_list))]))

                self.feature_map[field_v].update(d)
            else:
                field_list = list(
                    set(dataframe[field_k].drop_duplicates().values))
                field_list.sort()
                d = dict(zip(field_list, [i for i in range(len(field_list))]))

                self.feature_map[field_v] = d
            self.feature_size[field_v] = len(self.feature_map[field_v])

    def apply_feature(self, dataframe: DataFrame, fields: dict = None) -> DataFrame:
        """
        字段映射
        :param dataframe:
        :param fields:
        :return:
        """
        fields = self._field_converse(dataframe, fields)

        for field_k, field_v in fields.items():
            if field_v not in self.feature_map.keys():
                continue
            dataframe[field_k] = dataframe[field_k].apply(
                lambda x: self.feature_map[field_v][x])
        return dataframe

    def add_feature_and_apply(self, dataframe: DataFrame, fields: dict = None) -> DataFrame:
        self.add_feature(dataframe, fields)
        return self.apply_feature(dataframe, fields)
