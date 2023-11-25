import pyqtgraph as pg
import random
import numpy as np

randcolor = lambda:random.randint(0x10, 0xFF)

class BarGraph:
    merge_th = 0.9

    def __init__(self, layout, row, data : np.ndarray,
                       test_time : float, instances : int, bins : int, bin_sz : float, tavg : float,
                       title : str, instance_name : str,
                       display_time : float, frametime : float,
                       partial_fill : bool = True, colors = None,
                       ) -> None:
        frames = display_time/frametime
        self.len_step = test_time/frames

        self.plot = layout.addPlot(row = row, col = 0, colspan = 3, title = title)
        self.bars = pg.BarGraphItem(x = [0], y = [0], height = 1, width = [0], pen = None)
        self.cover = pg.BarGraphItem(x1 = test_time, width = test_time, y0 = 0, y1 = instances+1, brush = 'black', pen = None)
        self.line = pg.InfiniteLine(pos = 0, pen = 'red')

        self.plot.addItem(self.bars)
        self.plot.addItem(self.cover)
        self.plot.addItem(self.line)

        self.plot.setLabel('left', instance_name)
        self.plot.setLabel('bottom', 'Time', units = 's')

        self.histogram = layout.addPlot(row = row, col = 3, title = title)
        self.hist_bars = pg.BarGraphItem(x = np.arange(instances)+1, y0 = 0, height = 0, width = 1, pen = None)

        self.histogram.addItem(self.hist_bars)

        self.histogram.setLabel('left', 'Iterations', units = ' ')
        self.histogram.setLabel('bottom', instance_name)

        self.data = data
        self.data_2d = data.reshape((-1, instances))
        self.instances = instances
        self.bins = bins
        self.count = self.instances*self.bins
        self.bin_sz = bin_sz
        self.ns_per_ctr = tavg
        self.test_time = test_time

        self.partial_fill = partial_fill

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_bars)

        self.colors = colors

    def setup(self) -> None:
        self.line.setPos(0)

        self.cur_len = 0
        xstarts = np.repeat([b*self.bin_sz/1e9 for b in range(self.bins)], self.instances)

        if self.partial_fill:
            # Partially fill all
            lens = self.data*self.ns_per_ctr/1e9
        else:
            # Fully fill bins prior to end
            lens = np.where(self.data > 0, self.bin_sz/1e9, 0)

            # Partially fill last index
            lens[-self.instances:] = self.data[-self.instances:]*self.ns_per_ctr/1e9

        if self.colors is None:
            self.colors = [(randcolor(), randcolor(), randcolor()) for _ in range(self.instances)]

        # Merge consecutive blocks that are completely filled
        # to reduce the total number of polygons to paint
        kwargs = self.merge_bars(xstarts, lens)

        # Filter out any empty bins
        nz_idxes = np.where(kwargs['width'] > 1e-9)[0]
        kwargs = { 'x0'      : kwargs['x0'][nz_idxes],
                   'width'   : kwargs['width'][nz_idxes],
                   'y'       : kwargs['y'][nz_idxes],
                   'brushes' : kwargs['brushes'][nz_idxes],
                   }

        print(f'Zero Filter --> orig len: {len(lens)} | new len: {len(kwargs["width"])}')

        self.bars.setOpts(**kwargs)
        self.hist_bars.setOpts(brushes = self.colors)

    def merge_bars(self, xstarts : np.ndarray, lens : np.ndarray) -> dict:
        xstarts_2d = xstarts.reshape((-1, self.instances))
        lens_2d = lens.reshape((-1, self.instances))

        th = self.merge_th*self.bin_sz/1e9
        kwarg_list = []
        for inst_idx in range(self.instances):
            xstarts_col = xstarts_2d[:,inst_idx]
            lens_col = lens_2d[:,inst_idx]

            ctr = 0
            inst_kwargs = { 'x0'    : [xstarts_col[0]],
                            'width' : [lens_col[0]],
                            }
            for idx in range(1, len(lens_col)):
                if lens_col[idx-1] > th:
                    inst_kwargs['width'][ctr] = xstarts_col[idx] - inst_kwargs['x0'][ctr] + lens_col[idx]
                elif lens_col[idx] > 1e-9:
                    inst_kwargs['x0'].append(xstarts_col[idx])
                    inst_kwargs['width'].append(lens_col[idx])
                    ctr += 1
            # Make sure end does not exceed test time
            if (inst_kwargs['x0'][-1] + inst_kwargs['width'][-1]) > self.test_time:
                inst_kwargs['width'][-1] = self.test_time - inst_kwargs['x0'][-1]
            kwarg_list.append(inst_kwargs)

        x0_list = []
        width_list = []
        y_list = []
        brushes_list = []
        for i in range(self.instances):
            inst_len = len(kwarg_list[i]['x0'])
            x0_list += kwarg_list[i]['x0']
            width_list += kwarg_list[i]['width']
            y_list += [1+i]*inst_len
            brushes_list += [self.colors[i]]*inst_len

        print(f'Merge --> orig len: {len(lens)} | new len: {len(width_list)}')

        return { 'x0'      : np.array(x0_list),
                    'width'   : np.array(width_list),
                    'y'       : np.array(y_list),
                    'brushes' : np.array(brushes_list),
                    }

    def update_bars(self) -> None:
        self.cur_len += self.len_step

        if self.cur_len > self.test_time:
            self.timer.stop()
            return

        self.refresh()

    def refresh(self) -> None:
        cover_len = self.test_time-self.cur_len
        self.cover.setOpts(x1 = self.test_time, width = cover_len)
        self.line.setPos(self.cur_len)

        bin_idx = min(int(1e9*self.cur_len/self.bin_sz)+1, self.bins)
        self.hist_bars.setOpts(height = [np.sum(self.data_2d[:bin_idx,i]) for i in range(self.instances)])