from net.hr_module import HRModule


class HRStage():

    def __init__(self, stage_id, num_modules, num_branches, num_channels, num_blocks, last_stage=False):

        self.stage_id = stage_id
        self.num_modules = num_modules
        self.num_branches = num_branches
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.scope = 'HRStage_{}'.format(self.stage_id)
        self.hr_modules = []
        self.last_stage = last_stage
        self.build()

    def build(self):
        for i in range(self.num_modules):
            _hr = HRModule(module_id=i, num_branches=self.num_branches, num_blocks=self.num_blocks,
                           num_channels=self.num_channels,
                           multi_scale_output=self.get_multi_scale_output_flag(i),
                           scope=self.scope)
            self.hr_modules.append(_hr)

    def forward(self, inputs):
        output = inputs
        for hrmodule in self.hr_modules:
            output = hrmodule.forward(output)

        return output

    def get_multi_scale_output_flag(self, module_id):
        if self.last_stage:
            return False
        else:
            return True if module_id == self.num_modules - 1 else False
