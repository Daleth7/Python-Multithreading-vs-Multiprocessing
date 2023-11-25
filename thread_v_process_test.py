import time
from typing import List
from threading import Thread
from multiprocessing import Process, Array, Pool, Pipe
from multiprocessing.connection import Connection
import numpy as np
import pyqtgraph as pg

from BarGraph import BarGraph

instances = 4
test_time = 1 # s

bin_sz = 200000 # ns
bin_len = int(test_time*1e9/bin_sz)

partial_fill = True
frametime = 16 # ms
display_time = 15 # s

io_dly = 0.1 # s

run_threads    = True
run_proc       = False
run_pipe       = True
run_pool       = False
run_io_threads = False
run_io_pipe    = False

def task(tref : float, bin_ctr, bsz : float, task_idx : int):
    while True:
        delt = time.perf_counter_ns() - tref
        idx = int(delt/bsz)
        if idx >= bin_len:
            break
        bin_ctr[idx*instances+task_idx] += 1

def task_io(tref : float, bin_ctr, bsz : float, task_idx : int):
    while True:
        delt = time.perf_counter_ns() - tref
        idx = int(delt/bsz)
        if idx >= bin_len:
            break
        bin_ctr[idx*instances+task_idx] += 1
        time.sleep(io_dly)

def task_pipe(tref : float, conn : Connection, bsz : float) -> np.ndarray:
    bin_ctr = np.zeros(bin_len)
    while True:
        delt = time.perf_counter_ns() - tref
        idx = int(delt/bsz)
        if idx >= bin_len:
            break
        bin_ctr[idx] += 1
    conn.send(bin_ctr)

def task_io_pipe(tref : float, conn : Connection, bsz : float) -> np.ndarray:
    bin_ctr = np.zeros(bin_len)
    while True:
        delt = time.perf_counter_ns() - tref
        idx = int(delt/bsz)
        if idx >= bin_len:
            break
        bin_ctr[idx] += 1
        time.sleep(io_dly)
    conn.send(bin_ctr)

def task_pool(tref : float, bsz : float) -> np.ndarray:
    bin_ctr = np.zeros(bin_len)
    while True:
        delt = time.perf_counter_ns() - tref
        idx = int(delt/bsz)
        if idx >= bin_len:
            break
        bin_ctr[idx] += 1
    return bin_ctr

def calc_stats(bins, insts, counter, instance_name):
    bin_sum = 0.0
    nonzero_bin_count = 0
    for i in range(bins):
        idx = i*insts
        idx_end = (i+1)*insts
        task_sum = sum(counter[idx:idx_end])
        if task_sum > 0:
            bin_sum += task_sum
            nonzero_bin_count += 1
    delt = (tf-t0)
    tavg = bin_sz/(bin_sum/nonzero_bin_count/insts)
    print(f'{instance_name} results:')
    print(f'  Total execution time: {delt/1e6:.1f} ms')
    print(f'  Total iterations: {int(np.sum(counter)):,}')
    print(f'  Average non-zero bin value: {int(bin_sum/nonzero_bin_count/insts):,}')
    print(f'  Average iteration time: {tavg:.1f} ns')

    return tavg

class TogglePause:
    def __init__(self, graphs : List[BarGraph]) -> None:
        self.graphs = graphs
        self.paused = False

    def __call__(self) -> None:
        self.paused = not self.paused
        for g in self.graphs:
            if self.paused:
                g.timer.stop()
            else:
                g.timer.start(frametime)

class Rewinder:
    def __init__(self, graphs : List[BarGraph], pauser : TogglePause) -> None:
        self.graphs = graphs
        self.pauser = pauser

    def __call__(self) -> None:
        jump_scale = 3 if self.pauser.paused else 10
        for g in self.graphs:
            g.cur_len = max(g.cur_len-jump_scale*g.len_step, 0.0)
            g.refresh()

class Forwarder:
    def __init__(self, graphs : List[BarGraph], pauser : TogglePause) -> None:
        self.graphs = graphs
        self.pauser = pauser

    def __call__(self) -> None:
        jump_scale = 3 if self.pauser.paused else 10
        for g in self.graphs:
            g.cur_len = min(g.cur_len+jump_scale*g.len_step, test_time)
            g.refresh()

if __name__ == '__main__': # for multiprocessing

    if run_threads:
        print('Preparing threads...')
        thread_bin_counter = np.zeros(instances*bin_len)
        tref = time.perf_counter_ns()
        tlist = [Thread(target = task, args = (tref, thread_bin_counter, bin_sz, i)) for i in range(instances)]

        print('Collecting data with threads...')
        t0 = time.perf_counter_ns()

        for t in tlist:
            t.start()

        for t in tlist:
            t.join()

        tf = time.perf_counter_ns()
        print('Data collection complete!')
        thread_tavg_it = calc_stats(bin_len, instances, thread_bin_counter, 'Threads')
    else:
        thread_bin_counter = np.zeros(instances*bin_len)
        thread_tavg_it = 0.0



    if run_proc:
        print('Preparing processes (Shared Array)...')
        pbin_counter = Array('i', np.zeros(instances*bin_len, dtype = np.int64))
        tref = time.perf_counter_ns()
        plist = [Process(target = task, args = (tref, pbin_counter, bin_sz, i)) for i in range(instances)]

        print('Collecting data with processes (Shared Array)...')
        t0 = time.perf_counter_ns()

        for p in plist:
            p.start()

        for p in plist:
            p.join()

        tf = time.perf_counter_ns()
        print('Data collection complete!')
        pbin_counter = np.array(pbin_counter)
        proc_tavg_it = calc_stats(bin_len, instances, pbin_counter, 'Processes (Shared Array)')
    else:
        pbin_counter = np.zeros(instances*bin_len)
        proc_tavg_it = 0.0



    if run_pipe:
        print('Preparing processes (Pipe)...')
        conns = [Pipe() for _ in range(instances)]
        tref = time.perf_counter_ns()
        plist = [Process(target = task_pipe, args = (tref, conns[i][1], bin_sz)) for i in range(instances)]

        print('Collecting data with processes (Pipe)...')
        t0 = time.perf_counter_ns()

        for p in plist:
            p.start()

        pipe_res_list = [conns[i][0].recv() for i in range(instances)]

        for p in plist:
            p.join()

        tf = time.perf_counter_ns()
        print('Data collection complete!')
        pipe_bin_counter = np.concatenate(pipe_res_list).reshape((instances, -1)).transpose().flatten()
        pipe_tavg_it = calc_stats(bin_len, instances, pipe_bin_counter, 'Processes (Pipe)')
    else:
        pipe_bin_counter = np.zeros(instances*bin_len)
        proc_tavg_it = 0.0



    if run_pool:
        tref = time.perf_counter_ns()
        print('Collecting data with processes (Pool)...')
        t0 = time.perf_counter_ns()
        with Pool(processes = instances) as pool:
            pool_res_list = pool.starmap(task_pool, [(tref, bin_sz)]*instances)
        tf = time.perf_counter_ns()
        print('Data collection complete!')
        pool_bin_counter = np.concatenate(pool_res_list).reshape((instances, -1)).transpose().flatten()
        pool_tavg_it = calc_stats(bin_len, instances, pool_bin_counter, 'Processes (Pool)')
    else:
        pool_bin_counter = np.zeros(instances*bin_len)
        pool_tavg_it = 0.0



    if run_io_threads:
        print('Preparing threads (IO)...')
        thread_io_bin_counter = np.zeros(instances*bin_len)
        tref = time.perf_counter_ns()
        tlist = [Thread(target = task if i%2 else task_io, args = (tref, thread_io_bin_counter, bin_sz, i)) for i in range(instances)]

        print('Collecting data with threads (IO)...')
        t0 = time.perf_counter_ns()

        for t in tlist:
            t.start()

        for t in tlist:
            t.join()

        tf = time.perf_counter_ns()
        print('Data collection complete!')
        thread_io_tavg_it = calc_stats(bin_len, instances, thread_io_bin_counter, 'Threads (IO)')
    else:
        thread_io_bin_counter = np.zeros(instances*bin_len)
        thread_io_tavg_it = 0.0



    if run_io_pipe:
        print('Preparing processes (Pipe w/ IO)...')
        conns = [Pipe() for _ in range(instances)]
        tref = time.perf_counter_ns()
        plist = [Process(target = task_pipe if i%2 else task_io_pipe, args = (tref, conns[i][1], bin_sz)) for i in range(instances)]

        print('Collecting data with processes (Pipe w/ IO)...')
        t0 = time.perf_counter_ns()

        for p in plist:
            p.start()

        pipe_res_list = [conns[i][0].recv() for i in range(instances)]

        for p in plist:
            p.join()

        tf = time.perf_counter_ns()
        print('Data collection complete!')
        pipe_io_bin_counter = np.concatenate(pipe_res_list).reshape((instances, -1)).transpose().flatten()
        pipe_io_tavg_it = calc_stats(bin_len, instances, pipe_bin_counter, 'Processes (Pipe w/ IO)')
    else:
        pipe_io_bin_counter = np.zeros(instances*bin_len)
        proc_io_tavg_it = 0.0



    print('Preparing to display...')
    layout = pg.GraphicsLayoutWidget()

    graph_list = []

    if run_threads:
        thread_graph = BarGraph( layout, 0, thread_bin_counter,
                                 test_time, instances, bin_len, bin_sz, thread_tavg_it,
                                 title = 'Multi-Threading (CPU Intensive)', instance_name = 'Thread',
                                 display_time = display_time, frametime = frametime/1e3,
                                 partial_fill = partial_fill
                                 )
        thread_graph.setup()
        graph_list.append(thread_graph)

    if run_proc:
        process_graph = BarGraph( layout, 1, pbin_counter,
                                  test_time, instances, bin_len, bin_sz, proc_tavg_it,
                                  title = 'Multi-Processing (CPU Intensive) (Shared Array)', instance_name = 'Process',
                                  display_time = display_time, frametime = frametime/1e3,
                                  partial_fill = partial_fill, colors = thread_graph.colors
                                  )
        process_graph.setup()
        graph_list.append(process_graph)

    if run_pipe:
        pipe_graph = BarGraph( layout, 2, pipe_bin_counter,
                               test_time, instances, bin_len, bin_sz, pipe_tavg_it,
                               title = 'Multi-Processing (CPU Intensive) (Pipe)', instance_name = 'Process',
                               display_time = display_time, frametime = frametime/1e3,
                               partial_fill = partial_fill, colors = thread_graph.colors
                               )
        pipe_graph.setup()
        graph_list.append(pipe_graph)

    if run_pool:
        pool_graph = BarGraph( layout, 3, pool_bin_counter,
                               test_time, instances, bin_len, bin_sz, pool_tavg_it,
                               title = 'Multi-Processing (CPU Intensive) (Pool)', instance_name = 'Process',
                               display_time = display_time, frametime = frametime/1e3,
                               partial_fill = partial_fill, colors = thread_graph.colors
                               )
        pool_graph.setup()
        graph_list.append(pool_graph)

    if run_io_threads:
        thread_io_graph = BarGraph( layout, 4, thread_io_bin_counter,
                                    test_time, instances, bin_len, bin_sz, thread_io_tavg_it,
                                    title = 'Multi-Threading (IO Intensive)', instance_name = 'Thread',
                                    display_time = display_time, frametime = frametime/1e3,
                                    partial_fill = partial_fill, colors = thread_graph.colors
                                    )
        thread_io_graph.setup()
        graph_list.append(thread_io_graph)

    if run_io_pipe:
        pipe_io_graph = BarGraph( layout, 5, pipe_io_bin_counter,
                                  test_time, instances, bin_len, bin_sz, pipe_io_tavg_it,
                                  title = 'Multi-Processing (IO Intensive) (Pipe)', instance_name = 'Process',
                                  display_time = display_time, frametime = frametime/1e3,
                                  partial_fill = partial_fill, colors = thread_graph.colors
                                  )
        pipe_io_graph.setup()
        graph_list.append(pipe_io_graph)

    pause_toggler = TogglePause(graph_list)
    toggle_pause_shortcut = pg.QtGui.QShortcut(pg.QtGui.QKeySequence('Space'), layout)
    toggle_pause_shortcut.activated.connect(pause_toggler)

    rewinder = Rewinder(graph_list, pause_toggler)
    rewind_shortcut = pg.QtGui.QShortcut(pg.QtGui.QKeySequence('left'), layout)
    rewind_shortcut.activated.connect(rewinder)

    forwarder = Forwarder(graph_list, pause_toggler)
    forward_shortcut = pg.QtGui.QShortcut(pg.QtGui.QKeySequence('right'), layout)
    forward_shortcut.activated.connect(forwarder)

    lead_graph = graph_list[0]
    for graph in graph_list[1:]:
        graph.plot.setXLink(lead_graph.plot)
        graph.plot.setYLink(lead_graph.plot)

        graph.histogram.setXLink(lead_graph.histogram)

    for graph in graph_list:
        graph.timer.start(frametime)

    layout.show()

    pg.exec()