# A Redis gear for orchestrating realtime video analytics
import io
import cv2
import redisAI
import numpy as np
from time import time
from PIL import Image
from collections import OrderedDict, defaultdict

from redisgears import executeCommand as execute

# Globals for downsampling
_mspf = 1000 / 10.0      # Msecs per frame (initialized with 10.0 FPS)

_next_ts = 0#defaultdict(lambda: 0, {})             # Next timestamp to sample a frame

class SimpleMovingAverage(object):
    ''' Simple moving average '''
    def __init__(self, value=0.0, count=7):
        '''
        @value - the initialization value
        @count - the count of samples to keep
        '''
        self.count = int(count)
        self.current = float(value)
        self.samples = [self.current] * self.count

    def __str__(self):
        return str(round(self.current, 3))

    def add(self, value):
        ''' Adds the next value to the average '''
        v = float(value)
        self.samples.insert(0, v)
        o = self.samples.pop()
        self.current = self.current + (v-o)/self.count

class Profiler(object):
    ''' Mini profiler '''
    names = []  # Steps names in order
    data = {}   # ... and data
    last = None
    def __init__(self):
        pass

    def __str__(self):
        s = ''
        for name in self.names:
            s = '{}{}:{}, '.format(s, name, self.data[name])
        return(s[:-2])

    def __delta(self):
        ''' Returns the time delta between invocations '''
        now = time()*1000       # Transform to milliseconds
        if self.last is None:
            self.last = now
        value = now - self.last
        self.last = now
        return value

    def start(self, s_t=None):
        ''' Starts the profiler '''
        if s_t is None:
            s_t = time()*1000
        self.last = s_t

    def add(self, name):
        ''' Adds/updates a step's duration '''
        value = self.__delta()
        if name not in self.data:
            self.names.append(name)
            self.data[name] = SimpleMovingAverage(value=value)
        else:
            self.data[name].add(value)

    def assign(self, name, value):
        ''' Assigns a step with a value '''
        if name not in self.data:
            self.names.append(name)
            self.data[name] = SimpleMovingAverage(value=value)
        else:
            self.data[name].add(value)

    def get(self, name):
        ''' Gets a step's value '''
        return self.data[name].current

'''
The profiler is used first and foremost for keeping track of the total (average) time it takes to process
a frame - the information is required for setting the FPS dynamically. As a side benefit, it also provides
per step metrics.
'''
prf = Profiler()

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1])  # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0])  # y1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1])  # x2
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def letterbox_image(imgs, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''

    batch, img_h, img_w, cols = imgs.shape
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    canvas = np.full((batch, h, w, cols), 128, dtype=np.uint8)

    # Type 1
    # for idx, img in enumerate(imgs):
    #     resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    #     canvas[idx, (h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    # Type 2
    np_images = np.array([cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR) for img in imgs], dtype=np.uint8)
    canvas[:, (h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = np_images
    return canvas


def runYolo(X):
    ''' Runs the model on an input image from the stream '''
    global prf
    global IMG_SIZE    # Model's input image size

    # log('read')
    # log(f"keys {X.keys()}")
    # log(f"vals {X.values()}")
    # Read the image from the stream's message

    imgs = []
    ids = []
    prf.start(int(X.pop('ts', None)))        # Start a new profiler iteration
    s_keys = sorted(X.keys())

    for key in s_keys:
        img, _id = load_img(X[key]['image'], X[key]['shape']), X[key]['id']
        imgs.append(img)
        # log(str(imgs[-1].shape))
        ids.append(_id)

    numpy_imgs = np.array(imgs)
    # log(str(numpy_imgs.shape))
    orig_shape = numpy_imgs[0].shape
    prf.add('read')

    # Resize, normalize and tensorize the image for the model (number of images, width, height, channels)
    image_tensor = letterbox_image(numpy_imgs, (IMG_SIZE, IMG_SIZE))
    image_tensor = image_tensor.transpose(0, 3, 1, 2) / 255.0
    image_tensor = np.ascontiguousarray(image_tensor, dtype=np.float32)
    image_tensor = redisAI.createTensorFromBlob('FLOAT', image_tensor.shape, image_tensor.data)

    prf.add('resize')


    # log('resize')
    # Create the RedisAI model runner and run it
    modelRunner = redisAI.createModelRunner('yolo:model')
    redisAI.modelRunnerAddInput(modelRunner, 'input', image_tensor)
    redisAI.modelRunnerAddOutput(modelRunner, 'output')
    model_replies = redisAI.modelRunnerRun(modelRunner)
    model_output = model_replies[0]
    prf.add('model')
    # log(f"model result shape {redisAI.tensorGetDims(model_output)}")
    # log('model')
    # The model's output is processed with a PyTorch script for non maxima suppression
    scriptRunner = redisAI.createScriptRunner('yolo:script', 'boxes_from_yolo')
    redisAI.scriptRunnerAddInput(scriptRunner, model_output)
    [redisAI.scriptRunnerAddOutput(scriptRunner) for _, _ in enumerate(X.keys())]
    script_reply = redisAI.scriptRunnerRun(scriptRunner)
    prf.add('script')
    # log('script')

    all_results = []
    for idx, k in enumerate(X.keys()):
        # The script outputs bounding boxes
        shape = redisAI.tensorGetDims(script_reply[idx])
        buf = redisAI.tensorGetDataAsBlob(script_reply[idx])
        # Get boxes and re-scale them
        # log(str(shape))
        boxes = np.frombuffer(buf, dtype=np.float32).reshape(shape)
        boxes = scale_coords([IMG_SIZE, IMG_SIZE], boxes, orig_shape)

        # Iterate boxes to extract the people
        boxes_out = []
        people_count = 0
        for box in boxes:
            # if box[4] == 0.0:  # Remove zero-confidence detections
            #     continue
            # if box[-1] != 0:  # Ignore detections that aren't people
            #     continue
            people_count += 1
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]

            # Store boxes as a flat list
            boxes_out += [x1,y1,x2,y2]

        all_results.append((ids[idx], people_count, boxes_out))

    prf.add('boxes')
    return all_results[0]


def storeResults(x):
    ''' Stores the results in Redis Stream and TimeSeries data structures '''
    global _mspf, prf
    ref_id, people, boxes = x[0], int(x[1]), x[2]
    ref_msec = int(str(ref_id).split('-')[0])

    # Store the output in its own stream
    res_id = execute('XADD', 'camera_out:0:yolo', 'MAXLEN', '~', 1000, '*', 'ref', ref_id, 'boxes', boxes, 'people', people)

    # Add a sample to the output people and fps timeseries
    res_msec = int(str(res_id).split('-')[0])
    try:
        execute('TS.ADD', 'camera:0:people', ref_msec, people)
        execute('TS.INCRBY', 'camera:0:out_fps', 1)
    except:
        pass

    # Adjust mspf to the moving average duration
    total_duration = res_msec - ref_msec
    prf.assign('total', total_duration)
    avg_duration = prf.get('total')
    _mspf = avg_duration * 1.05  # A little extra leg room

    # Record profiler steps
    for name in prf.names:
        current = prf.data[name].current

        try:
            execute('TS.ADD', 'camera_in:0:prf_{}'.format(name), ref_msec, current)
        except:
            log(f'name: {name}')
            log(f'current: {current}')
            log(f'ref_msec: {ref_msec}')
            log(' ')

    prf.add('store')
    # Make an arithmophilial homage to Count von Count for storage in the execution log
    if people == 0:
        return 'Now there are none.'
    elif people == 1:
        return 'There is one person in the frame!'
    elif people == 2:
        return 'And now there are are two!'
    else:
        return 'I counted {} people in the frame! Ah ah ah!'.format(people)


def downsampleStream(x):
    ''' Drops input frames to match FPS '''
    global _mspf, _next_ts

    try:
        execute('TS.INCRBY', x['camera_in:0']['key']+':in_fps', 1)  # Store the input fps count
    except:
        pass
    ts = x['ts'] # Extract the timestamp part from the message ID
    sample_it = _next_ts <= ts
    if sample_it:                             # Drop frames until the next timestamp is in the present/past
        _next_ts = ts + _mspf
    return sample_it


def frameAccumulator(a, r):
    a = a if a else OrderedDict()
    camera, msg_id = r['value']['cam'], r['id']
    a[camera] = {'image': r['value']['image'], 'id': msg_id, 'key': r['key'], 'shape': r['value']['shape']}
    r_ts = int(msg_id.split('-')[0])
    a['ts'] = min(r_ts, a.get('ts', r_ts))
    return a

# def multiplexer(x):
#     # log(str(x['value']['image']))
#     execute('XADD', f'batchStream{{{hashtag()}}}', '*', x['key'], x['value']['image'], 'id', x['id'])


def load_img(x, shp):
    global IMG_SIZE
    if not isinstance(x, np.ndarray):
        try:
            # Type 1 - 8ms avg - encode\decode
            # buf = io.BytesIO(x)
            # opn = Image.open(buf)
            # im = np.array(opn)

            # Type 2 - 1ms< avg - sending bytes
            im = np.frombuffer(x, dtype=np.uint8).reshape(eval(shp))
        except:
            im = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    else:
        im = x

    return im


def padBatch(x):
    global BATCH_SIZE
    some_item = list(x.values())[0]
    an_id, a_key, a_shape = some_item['id'], some_item['key'], some_item['shape']
    # log(f"it took {1000*time()-x['ts']:.3f}ms to accumulate {len(x.values())-1}")
    if len(x.keys()) < BATCH_SIZE + 1:
        for k in [f'camera_in:{i}' for i in range(BATCH_SIZE)]:
            if k not in x.keys():
                x[k] = {'image': np.zeros(eval(a_shape), dtype=np.uint8), 'id': an_id, 'key': a_key, 'shape': a_shape}
    return x



# # Create and register a gear that for each message in the stream
# gb_mux = GearsBuilder('StreamReader')
# gb_mux.map(multiplexer)
# gb_mux.register('camera_in:*')

IMG_SIZE = 416
BATCH_SIZE = 8

gb = GearsBuilder('StreamReader')
gb.accumulate(frameAccumulator) # Accumulate single frames into a batch
gb.map(padBatch)                # Pad to required size if necessary
gb.filter(downsampleStream)     # Filter out high frame rate
gb.map(runYolo)                 # Run the model
gb.map(storeResults)            # Store the results
gb.register('batchStream*', batch=BATCH_SIZE)
