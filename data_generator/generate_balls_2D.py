import multiprocessing as mp
import time
import os
import cv2
import numpy as np
import random
import pymunk
from pymunk.vec2d import Vec2d
from progressbar import ProgressBar
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

from utils import rand_float, rand_int, calc_dis, norm, read_config
from utils import init_stat, combine_stat, load_data, store_data
from utils import resize, crop

cfg = read_config()



class Generator:
    def __init__(self, dir_out, seed, nb_examples):
        self.dir_out = dir_out
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        self.list_time = []
        self.nb_examples = nb_examples

        self.total_trial_counter = 0
        self.ab_trial_counter = 0
        self.cd_trial_counter = 0

    def generate(self):
        # if the data hasn't been generated, generate the data
        n_rollout, time_step, dt = self.n_rollout, self.args.time_step, self.args.dt
        assert n_rollout % self.args.num_workers == 0

        print("Generating data ... n_rollout=%d, time_step=%d" % (n_rollout, time_step))

        # 多进程，此处num_workers值一般为10
        # 根据传入参数来确定 等下要传给具体gen方法的参数
        infos = []
        # 为每个进程设置不同参数
        for i in range(self.args.num_workers):
            info = {'thread_idx': i,
                    'data_dir': self.data_dir,
                    'data_names': self.data_names,
                    'n_rollout': n_rollout // self.args.num_workers,
                    'time_step': time_step,
                    'dt': dt,
                    'video': False,
                    'phase': self.phase,
                    'args': self.args,
                    'vis_height': self.args.height_raw,
                    'vis_width': self.args.width_raw}

            if self.args.env in ['Ball']:
                info['env'] = 'Ball'
                info['n_ball'] = self.args.n_ball

            infos.append(info)

        cores = self.args.num_workers
        pool = mp.Pool(processes=cores)

        env = self.args.env

        # 调用上面的gen_{env}方法，生成数据data
        if env in ['Ball']:
            data = pool.map(self.gen_Ball, infos)
        else:
            raise AssertionError("Unknown env")

        print("Training data generated, warpping up stats ...")

        if env in ['Ball']:
            self.stat = [init_stat(self.args.attr_dim),
                         init_stat(self.args.state_dim),
                         init_stat(self.args.action_dim)]

        # 即依次融合，求整个的stat
        # 这里的data：
        # [视频1[mean, std, cnt]，视频2[mean, std, cnt]，...]
        for i in range(len(data)):
            for j in range(len(self.stat)):
                self.stat[j] = combine_stat(self.stat[j], data[i][j])

        # 保存stat
        store_data(self.data_names[:len(self.stat)], self.stat, self.stat_path)

    # 生成小球的数据集
    def gen_Ball(info):
        thread_idx, data_dir, data_names = info['thread_idx'], info['data_dir'], info['data_names']
        n_rollout, time_step = info['n_rollout'], info['time_step']
        dt, video, args, phase = info['dt'], info['video'], info['args'], info['phase']
        n_ball = info['n_ball']

        np.random.seed(round(time.time() * 1000 + thread_idx) % 2 ** 32)

        attr_dim = args.attr_dim  # radius
        state_dim = args.state_dim  # x, y, xdot, ydot
        action_dim = 2  # ddx, ddy

        # 对于该进程负责的所有rollout里，所有时间步的statistics
        # mean, std, count
        # stats: [attr, state, action]
        # attr: [ [attr1_mean, attr1_std, attr1_count], [attr2...], ...]
        stats = [init_stat(attr_dim), init_stat(state_dim), init_stat(action_dim)]

        engine = BallEngine(dt, state_dim, action_dim=2)

        bar = ProgressBar()
        for i in bar(range(n_rollout)):
            rollout_idx = thread_idx * n_rollout + i
            rollout_dir = os.path.join(data_dir, str(rollout_idx))
            os.system('mkdir -p ' + rollout_dir)

            engine.init(n_ball)

            # 小球的数量
            n_obj = engine.num_obj

            # 时间步 * 物体 * 属性维数
            attrs_all = np.zeros((time_step, n_obj, attr_dim))
            # 时间步 * 物体 * 状态维数
            states_all = np.zeros((time_step, n_obj, state_dim))
            # 时间步 * 物体 * 动作维数
            actions_all = np.zeros((time_step, n_obj, action_dim))
            # 时间步 * {param_dim = n_ball * (n_ball - 1)} * 2
            # param_dim即两两之间的关系：[relation_type, coefficient]
            rel_attrs_all = np.zeros((time_step, engine.param_dim, 2))

            act = np.zeros((n_obj, 2))
            # 对每个时间步进行迭代
            for j in range(time_step):
                # 当前状态
                state = engine.get_state()
                # 整数除法
                vel_dim = state_dim // 2
                # 前vel_dim维：位置
                pos = state[:, :vel_dim]
                # 后面的维数：速度（velocity）
                vel = state[:, vel_dim:]

                # 通过两次的位置信息，计算此刻的速度
                if j > 0:
                    vel = (pos - states_all[j - 1, :, :vel_dim]) / dt

                # 计算各属性，状态信息，关系
                attrs = np.zeros((n_obj, attr_dim))
                attrs[:] = engine.radius

                attrs_all[j] = attrs
                states_all[j, :, :vel_dim] = pos
                states_all[j, :, vel_dim:] = vel
                rel_attrs_all[j] = engine.param

                # 带有随机噪声的力 用于下一步迭代
                act += (np.random.rand(n_obj, 2) - 0.5) * 600 - act * 0.1 - state[:, 2:] * 0.1
                act = np.clip(act, -1000, 1000)
                engine.step(act)

                actions_all[j] = act.copy()

            datas = [attrs_all, states_all, actions_all, rel_attrs_all]
            # stat.h5 存储上述数据data：[attrs_all, states_all, actions_all, rel_attrs_all]
            store_data(data_names, datas, rollout_dir + '.h5')

            # 渲染，导出至图片帧或视频
            engine.render(states_all, actions_all, engine.get_param(), video=False, image=True,
                          path=rollout_dir, draw_edge=False, verbose=False)

            # 转换为float64
            datas = [datas[i].astype(np.float64) for i in range(len(datas))]

            # 对物体属性，物体状态，action，三者分别求解均值，标准差，元素数量
            # stats: [ 物体属性, 物体状态, action]
            # 物体属性：[ 属性1, 属性2, ...]
            # 属性1: [mean, std, 时间步的数量(即这是多少个状态在求平均)]
            for j in range(len(stats)):
                stat = init_stat(stats[j].shape[0])
                # 所有时间步下，所有物体的统计量
                stat[:, 0] = np.mean(datas[j], axis=(0, 1))[:]
                stat[:, 1] = np.std(datas[j], axis=(0, 1))[:]
                stat[:, 2] = datas[j].shape[0]
                stats[j] = combine_stat(stats[j], stat)

        return stats


class Engine(object):

    def __init__(self, dt, state_dim, action_dim):
        self.dt = dt
        self.state_dim = state_dim  # 状态维度
        self.action_dim = action_dim  # 动作维度

        self.param_dim = None
        self.state = None
        self.action = None
        self.param = None

    def init(self):
        pass

    def get_param(self):
        return self.param.copy()

    def set_param(self, param):
        self.param = param.copy()

    def get_state(self):
        return self.state.copy()

    def set_state(self, state):
        self.state = state.copy()

    def get_scene(self):
        return self.state.copy(), self.param.copy()

    def set_scene(self, state, param):
        self.state = state.copy()
        self.param = param.copy()

    def get_action(self):
        return self.action.copy()

    def set_action(self, action):
        self.action = action.copy()

    def d(self, state, t, param):
        # time derivative
        pass

    def step(self):
        pass

    def render(self, state, param):
        pass

    def clean(self):
        pass


class BallEngine(Engine):

    def __init__(self, dt, state_dim, action_dim):

        # state_dim = 4
        # action_dim = 2
        # param_dim = n_ball * (n_ball - 1)

        # param [relation_type, coefficient]
        # relation_type
        # 0 - no relation
        # 1 - spring (DampedSpring)
        # 2 - string (SlideJoint)
        # 3 - rod (PinJoint)

        super(BallEngine, self).__init__(dt, state_dim, action_dim)

        self.init()

    # 添加线段
    def add_segments(self, p_range=(-80, 80, -80, 80)):
        # 静态线段
        # Segment(body: Body | None, a: Tuple[float, float], b: Tuple[float, float], radius: float)
        a = pymunk.Segment(self.space.static_body, (p_range[0], p_range[2]), (p_range[0], p_range[3]), 1)
        b = pymunk.Segment(self.space.static_body, (p_range[0], p_range[2]), (p_range[1], p_range[2]), 1)
        c = pymunk.Segment(self.space.static_body, (p_range[1], p_range[3]), (p_range[0], p_range[3]), 1)
        d = pymunk.Segment(self.space.static_body, (p_range[1], p_range[3]), (p_range[1], p_range[2]), 1)
        a.friction = 1;
        a.elasticity = 1  # 摩擦系数1 弹性系数1：完全弹性碰撞
        b.friction = 1;
        b.elasticity = 1
        c.friction = 1;
        c.elasticity = 1
        d.friction = 1;
        d.elasticity = 1
        self.space.add(a);
        self.space.add(b)
        self.space.add(c);
        self.space.add(d)

    # 添加所有小球
    def add_balls(self, center=(0., 0.), p_range=(-60, 60)):
        # 计算空心圆的转动惯量
        inertia = pymunk.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        # 添加所有小球
        # 每一次添加时，为新的小球随机选择一个位置，要求其距离已有小球的距离 >= 30
        for i in range(self.n_ball):
            while True:
                x = rand_float(p_range[0], p_range[1])
                y = rand_float(p_range[0], p_range[1])
                flag = True
                for j in range(i):
                    if calc_dis([x, y], self.balls[j].position) < 30:
                        flag = False
                if flag:
                    break
            # 添加新的刚体
            body = pymunk.Body(self.mass, inertia)
            body.position = Vec2d((x, y))
            # 其实所谓的小球就是质点
            # Circle(body: Body | None, radius: float, offset: Tuple[float, float] = (0, 0))
            shape = pymunk.Circle(body, 0., (0, 0))
            shape.elasticity = 1
            # 加入空间
            self.space.add(body, shape)
            self.balls.append(body)

    # 添加约束
    def add_rels(self, param_load=None):
        # param记录每一对关系
        # 每一对关系的要素: [relation_type, coefficient]
        param = np.zeros((self.n_ball * (self.n_ball - 1) // 2, 2))
        self.param_dim = param.shape[0]

        if param_load is not None:
            print("Load param for init env")

        cnt = 0
        rels_idx = []  # 将两两间的关系数组展平后的编号
        # 对于所有小球
        for i in range(self.n_ball):
            # 与先前的所有小球
            for j in range(i):
                # 选择并记录关系类型：随机/加载
                rel_type = rand_int(0, self.n_rel_type) if param_load is None else param_load[cnt, 0]
                param[cnt, 0] = rel_type

                rels_idx.append([i, j])

                pos_i = self.balls[i].position
                pos_j = self.balls[j].position

                if rel_type == 0:
                    # no relation
                    pass

                # 弹簧
                elif rel_type == 1:
                    # spring
                    # 剩余长度
                    rest_length = rand_float(20, 120) if param_load is None else param_load[cnt, 1]
                    param[cnt, 1] = rest_length
                    # DampedSpring(a: Body, b: Body, anchor_a: Tuple[float, float], /
                    # anchor_b: Tuple[float, float], rest_length: float, stiffness: float, damping: float)
                    # 弹性系数20，阻尼0
                    # 固定点即为小球质心
                    c = pymunk.DampedSpring(
                        self.balls[i], self.balls[j], (0, 0), (0, 0),
                        rest_length=rest_length, stiffness=20, damping=0.)
                    self.space.add(c)

                # 滑槽（有最大最小距离限制）
                elif rel_type == 2:
                    # string
                    rest_length = calc_dis(pos_i, pos_j) if param_load is None else param_load[cnt, 1]
                    param[cnt, 1] = rest_length
                    # SlideJoint(a: Body, b: Body, anchor_a: Tuple[float, float],
                    # anchor_b: Tuple[float, float], min: float, max: float)
                    # 滑槽的长度范围限制为-5到+5
                    c = pymunk.SlideJoint(
                        self.balls[i], self.balls[j], (0, 0), (0, 0),
                        rest_length - 5, rest_length + 5)
                    self.space.add(c)

                else:
                    raise AssertionError("Unknown relation type")

                cnt += 1

        if param_load is not None:
            assert ((param == param_load).all())

        self.rels_idx = rels_idx
        self.param = param

    # 冲量
    def add_impulse(self, p_range=(-200, 200)):
        # 对于所有小球
        for i in range(self.n_ball):
            # 直接在质心 施加随机的二维冲量
            impulse = (rand_float(p_range[0], p_range[1]), rand_float(p_range[0], p_range[1]))
            self.balls[i].apply_impulse_at_local_point(impulse=impulse, point=(0, 0))

    def add_boundary_impulse(self, p_range=(-75, 75, -75, 75)):
        """
        根据小球到边界的距离施加冲量
        通过检测小球距离预定边界的距离，根据距离的倒数给小球添加一个与距离成比例的冲量，以模拟小球与边界交互的物理效果。
        @param p_range position_range
        """
        f_scale = 5e2
        eps = 2
        for i in range(self.n_ball):
            impulse = np.zeros(2)
            # p：position
            p = np.array([self.balls[i].position[0], self.balls[i].position[1]])

            #
            d = min(20, max(eps, p[0] - p_range[0]))
            impulse[0] += f_scale / d
            d = max(-20, min(-eps, p[0] - p_range[1]))
            impulse[0] += f_scale / d
            d = min(20, max(eps, p[1] - p_range[2]))
            impulse[1] += f_scale / d
            d = max(-20, min(-eps, p[1] - p_range[3]))
            impulse[1] += f_scale / d

            self.balls[i].apply_impulse_at_local_point(impulse=impulse, point=(0, 0))

    def init(self, n_ball=5, init_impulse=True, param_load=None):
        # 定义空间
        self.space = pymunk.Space()
        # 水平和竖直都无重力作用
        self.space.gravity = (0., 0.)
        # 关系种类
        self.n_rel_type = 3
        # 小球数量
        self.n_ball = n_ball
        # 质量
        self.mass = 1.
        # 半径
        self.radius = 6
        self.balls = []
        # self.add_segments()
        # 添加所有小球
        self.add_balls()
        # 随机指定关系
        self.add_rels(param_load)
        # 添加初始冲量
        if init_impulse:
            self.add_impulse()

        self.state_prv = None

    # get小球数量
    @property
    def num_obj(self):
        return self.n_ball

    # 获取所有小球的状态
    # 状态：[posX, posY, vx, vy]
    # 其中速度v是平均速度
    def get_state(self):
        # state: [ball1, ball2, ...]
        # ball1_i: [posX, posY, vx, vy]
        state = np.zeros((self.n_ball, 4))
        for i in range(self.n_ball):
            ball = self.balls[i]
            state[i] = np.array([ball.position[0], ball.position[1], ball.velocity[0], ball.velocity[1]])

        vel_dim = self.state_dim // 2
        # 计算平均速度
        if self.state_prv is None:
            state[:, vel_dim:] = 0
        else:
            state[:, vel_dim:] = (state[:, :vel_dim] - self.state_prv[:, :vel_dim]) / self.dt

        return state

    # 在所有小球上按action逐个添加力
    # action: [force1, force2, ...]
    # force: [Fx, Fy]
    def add_action(self, action):
        if action is None:
            return
        for i in range(self.n_ball):
            # apply_force_at_local_point(force: Tuple[float, float], point: Tuple[float, float] = (0, 0))
            self.balls[i].apply_force_at_local_point(force=action[i], point=(0, 0))

    # 使用给定的力action进行一轮迭代
    def step(self, action=None):
        self.state_prv = self.get_state()
        self.add_action(action)
        self.add_boundary_impulse()
        self.space.step(self.dt)

    def render(self, states, actions, param, video=True, image=False, path=None, draw_edge=True,
               lim=(-80, 80, -80, 80), verbose=True, st_idx=0, image_prefix='fig'):
        # states: time_step x n_ball x 4
        # actions: time_step x n_ball x 2

        # lim = (lim[0] - self.radius, lim[1] + self.radius, lim[2] - self.radius, lim[3] + self.radius)

        # 导出视频
        if video:
            video_path = path + '.avi'
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            if verbose:
                print('Save video as %s' % video_path)
            out = cv2.VideoWriter(video_path, fourcc, 25, (110, 110))

        # 导出图像
        if image:
            image_path = path
            if verbose:
                print('Save images to %s' % image_path)
            command = 'mkdir -p %s' % image_path
            os.system(command)

        c = ['royalblue', 'tomato', 'limegreen', 'orange', 'violet', 'chocolate', 'black', 'crimson']

        time_step = states.shape[0]
        n_ball = states.shape[1]

        for i in range(time_step):
            fig, ax = plt.subplots(1)
            plt.xlim(lim[0], lim[1])
            plt.ylim(lim[2], lim[3])
            # plt.axis('off')

            fig.set_size_inches(1.5, 1.5)

            if draw_edge:
                # draw force
                for x in range(n_ball):
                    F = actions[i, x]

                    normF = norm(F)
                    Fx = F / normF * normF * 0.05
                    st = states[i, x, :2] + F / normF * 12.
                    ax.arrow(st[0], st[1], Fx[0], Fx[1], fc='Orange', ec='Orange', width=3., head_width=15.,
                             head_length=15.)

                # draw edge
                cnt = 0
                # 绘制两两小球之间的边
                for x in range(n_ball):
                    for y in range(x):
                        rel_type = int(param[cnt, 0]);
                        cnt += 1
                        if rel_type == 0:
                            continue

                        plt.plot([states[i, x, 0], states[i, y, 0]],
                                 [states[i, x, 1], states[i, y, 1]],
                                 '-', color=c[rel_type], lw=1, alpha=0.5)

            circles = []
            circles_color = []
            # 绘制代表小球的圆形
            for j in range(n_ball):
                circle = Circle((states[i, j, 0], states[i, j, 1]), radius=self.radius)
                circles.append(circle)
                circles_color.append(c[j % len(c)])

            pc = PatchCollection(circles, facecolor=circles_color, linewidth=0, alpha=0.5)
            ax.add_collection(pc)

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.tight_layout()

            if video or image:
                fig.canvas.draw()
                frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = frame[21:-19, 21:-19]

            if video:
                out.write(frame)
                if i == time_step - 1:
                    for _ in range(5):
                        out.write(frame)

            if image:
                cv2.imwrite(os.path.join(image_path, '%s_%s.png' % (image_prefix, i + st_idx)), frame)

            plt.close()

        if video:
            out.release()


if __name__ == '__main__':
    g = Generator(dir_out=args.dir_out, seed=args.seed, nb_examples=args.n_examples)
    g.generate()
