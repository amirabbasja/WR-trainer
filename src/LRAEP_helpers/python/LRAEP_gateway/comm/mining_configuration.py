from LRAEP_trainer.primitives import MiningConfiguration, MMAType, MoEConfig, PeriodicPattern

class LRAEPMiningConfigurationFactory:

    @classmethod
    def create(cls, common_dim: int, rank: int, row_indices: list[int], col_indices: list[int], moe: MoEConfig | None=None) -> MiningConfiguration:
        row_offset = min(row_indices)
        col_offset = min(col_indices)
        rows_pattern = PeriodicPattern.from_list([i - row_offset for i in row_indices])
        cols_pattern = PeriodicPattern.from_list([i - col_offset for i in col_indices])
        return MiningConfiguration(common_dim=common_dim, rank=rank, mma_type=MMAType.Int7xInt7ToInt32, rows_pattern=rows_pattern, cols_pattern=cols_pattern, moe=moe)
__all__ = ['MiningConfiguration', 'MMAType', 'MoEConfig', 'PeriodicPattern', 'LRAEPMiningConfigurationFactory']
