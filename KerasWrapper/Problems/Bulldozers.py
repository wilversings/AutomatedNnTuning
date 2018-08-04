from typing import List

from sklearn.preprocessing import LabelEncoder

from KerasWrapper.Problems.ProblemBase import ProblemBase, ProblemType
import numpy as np
import csv
from dateutil import parser

"""
P.4
"""
class Bulldozers(ProblemBase):

    INPUT_SIZE = 51
    OUTPUT_SIZE = 1
    PROB_TYPE = ProblemType.Regression

    def __init__(self, uri):
        cols = ["SalesId", "SalePrice", "MachineId", "ModelId", "Datasource", "AuctioneerId", "YearMade", "MachineHoursCurrentMeter", "UsageBand",
                "Saledate", "FiModelDesc", "FiBaseModel", "FiSecondaryDesc", "FiModelSeries", "FiModelDescriptor", "ProductSize", "FiProductClassDesc",
                "State", "ProductGroup", "ProductGroupDesc", "DriveSystem", "Enclosure", "Forks", "PadType", "RideControl", "Stick", "Transmission", "Turbocharged",
                "BladeExtension", "BladeWidth", "EnclosureType", "EngineHorsepower", "Hydraulics", "Pushblock", "Ripper", "Scarifier", "TipControl", "TireSize",
                "Coupler", "CouplerSystem", "GrouserTracks", "HydraulicsFlow", "TrackType", "UndercarriagePadWidth", "StickLength", "Thumb", "PatternChanger", "GrouserType"
                "BlackhoeMounting", "BladeType", "TravelControls", "DifferentialType", "SteeringControls"]
        cols_idx = list(range(len(cols)))


        for i in cols_idx:
            setattr(self, cols[i], i)

        self.Integers = cols_idx[:self.UsageBand]
        self.DateTimes = [cols_idx[self.Saledate]]
        self.Strings = [cols_idx[self.MachineHoursCurrentMeter]] + cols_idx[self.FiModelDesc:]

        super().__init__(uri)

        self._logger.info("Starting new population to test: P.4")


    def _load_data(self, uri):
        with open(uri) as inp:
            reader = csv.reader(inp, delimiter=',', quotechar='"')
            # Ignore the header of the CSV
            next(reader, None)

            recs = np.array([
                [float(val) if val != '' else -1 for val in line[:self.UsageBand]] +
                [line[self.UsageBand]] +
                [parser.parse(line[self.Saledate]).timestamp()] +
                line[self.FiModelDesc:] for line in reader], dtype='O')

        inp = np.concatenate(
            (recs[:, [x for x in self.Integers + self.DateTimes if x != self.SalePrice]], self.normalize_cols(recs[:, self.Strings])),
            axis=1)
        outp = recs[:, self.SalePrice]

        return list(zip(np.array(inp, dtype='float64'), np.array(outp, dtype='float64')))

    def normalize_cols(self, matrix):
        le = LabelEncoder()
        return np.transpose(
            list(map(le.fit_transform, np.transpose(matrix)))
        )

    @staticmethod
    def val_to_out_layer(val: str) -> List[int]:
        if val == "objective":
            return [1, 0]
        elif val == "subjective":
            return [0, 1]
        assert False