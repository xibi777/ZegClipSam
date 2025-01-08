import os.path as osp
from mmseg.datasets.builder import DATASETS
# from mmseg.datasets.custom import CustomDataset
from .mycustom import MyCustomDataset
import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg.utils import get_root_logger
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.pipelines import compose
from .loading import LoadAnnotationsTif
from PIL import Image


@DATASETS.register_module()
class MyADE20KDataset847(MyCustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K-847.
    """
    CLASSES = (
        'wall', 'building, edifice', 'sky', 'tree', 'road, route', 'floor, flooring', 'ceiling', 'bed', 'sidewalk, pavement', 'earth, ground', 'cabinet', 'person, individual, someone, somebody, mortal, soul', 'grass', 'windowpane, window', 'car, auto, automobile, machine, motorcar', 'mountain, mount', 'plant, flora, plant life', 'table', 'chair', 'curtain, drape, drapery, mantle, pall', 'door', 'sofa, couch, lounge', 'sea', 'painting, picture', 'water', 'mirror', 'house', 'rug, carpet, carpeting', 'shelf', 'armchair', 'fence, fencing', 'field', 'lamp', 'rock, stone', 'seat', 'river', 'desk', 'bathtub, bathing tub, bath, tub', 'railing, rail', 'signboard, sign', 'cushion', 'path', 'work surface', 'stairs, steps', 'column, pillar', 'sink', 'wardrobe, closet, press', 'snow', 'refrigerator, icebox', 'base, pedestal, stand', 'bridge, span', 'blind, screen', 'runway', 'cliff, drop, drop-off', 'sand', 'fireplace, hearth, open fireplace', 'pillow', 'screen door, screen', 'toilet, can, commode, crapper, pot, potty, stool, throne', 'skyscraper', 'grandstand, covered stand', 'box', 'pool table, billiard table, snooker table', 'palm, palm tree', 'double door', 'coffee table, cocktail table', 'counter', 'countertop', 'chest of drawers, chest, bureau, dresser', 'kitchen island', 'boat', 'waterfall, falls', 'stove, kitchen stove, range, kitchen range, cooking stove', 'flower', 'bookcase', 'controls', 'book', 'stairway, staircase', 'streetlight, street lamp', 'computer, computing machine, computing device, data processor, electronic computer, information processing system', 'bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle', 'swivel chair', 'light, light source', 'bench', 'case, display case, showcase, vitrine', 'towel', 'fountain', 'embankment', 'television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box', 'van', 'hill', 'awning, sunshade, sunblind', 'poster, posting, placard, notice, bill, card', 'truck, motortruck', 'airplane, aeroplane, plane', 'pole', 'tower', 'court', 'ball', 'aircraft carrier, carrier, flattop, attack aircraft carrier', 'buffet, counter, sideboard', 'hovel, hut, hutch, shack, shanty', 'apparel, wearing apparel, dress, clothes', 'minibike, motorbike', 'animal, animate being, beast, brute, creature, fauna', 'chandelier, pendant, pendent', 'step, stair', 'booth, cubicle, stall, kiosk', 'bicycle, bike, wheel, cycle', 'doorframe, doorcase', 'sconce', 'pond', 'trade name, brand name, brand, marque', 'bannister, banister, balustrade, balusters, handrail', 'bag', 'traffic light, traffic signal, stoplight', 'gazebo', 'escalator, moving staircase, moving stairway', 'land, ground, soil', 'board, plank', 'arcade machine', 'eiderdown, duvet, continental quilt', 'bar', 'stall, stand, sales booth', 'playground', 'ship', 'ottoman, pouf, pouffe, puff, hassock', 'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin', 'bottle', 'cradle', 'pot, flowerpot', 'conveyer belt, conveyor belt, conveyer, conveyor, transporter', 'train, railroad train', 'stool', 'lake', 'tank, storage tank', 'ice, water ice', 'basket, handbasket', 'manhole', 'tent, collapsible shelter', 'canopy', 'microwave, microwave oven', 'barrel, cask', 'dirt track', 'beam', 'dishwasher, dish washer, dishwashing machine', 'plate', 'screen, crt screen', 'ruins', 'washer, automatic washer, washing machine', 'blanket, cover', 'plaything, toy', 'food, solid food', 'screen, silver screen, projection screen', 'oven', 'stage', 'beacon, lighthouse, beacon light, pharos', 'umbrella', 'sculpture', 'aqueduct', 'container', 'scaffolding, staging', 'hood, exhaust hood', 'curb, curbing, kerb', 'roller coaster', 'horse, equus caballus', 'catwalk', 'glass, drinking glass', 'vase', 'central reservation', 'carousel', 'radiator', 'closet', 'machine', 'pier, wharf, wharfage, dock', 'fan', 'inflatable bounce game', 'pitch', 'paper', 'arcade, colonnade', 'hot tub', 'helicopter', 'tray', 'partition, divider', 'vineyard', 'bowl', 'bullring', 'flag', 'pot', 'footbridge, overcrossing, pedestrian bridge', 'shower', 'bag, traveling bag, travelling bag, grip, suitcase', 'bulletin board, notice board', 'confessional booth', 'trunk, tree trunk, bole', 'forest', 'elevator door', 'laptop, laptop computer', 'instrument panel', 'bucket, pail', 'tapestry, tapis', 'platform', 'jacket', 'gate', 'monitor, monitoring device', 'telephone booth, phone booth, call box, telephone box, telephone kiosk', 'spotlight, spot', 'ring', 'control panel', 'blackboard, chalkboard', 'air conditioner, air conditioning', 'chest', 'clock', 'sand dune', 'pipe, pipage, piping', 'vault', 'table football', 'cannon', 'swimming pool, swimming bath, natatorium', 'fluorescent, fluorescent fixture', 'statue', 'loudspeaker, speaker, speaker unit, loudspeaker system, speaker system', 'exhibitor', 'ladder', 'carport', 'dam', 'pulpit', 'skylight, fanlight', 'water tower', 'grill, grille, grillwork', 'display board', 'pane, pane of glass, window glass', 'rubbish, trash, scrap', 'ice rink', 'fruit', 'patio', 'vending machine', 'telephone, phone, telephone set', 'net', 'backpack, back pack, knapsack, packsack, rucksack, haversack', 'jar', 'track', 'magazine', 'shutter', 'roof', 'banner, streamer', 'landfill', 'post', 'altarpiece, reredos', 'hat, chapeau, lid', 'arch, archway', 'table game', 'bag, handbag, pocketbook, purse', 'document, written document, papers', 'dome', 'pier', 'shanties', 'forecourt', 'crane', 'dog, domestic dog, canis familiaris', 'piano, pianoforte, forte-piano', 'drawing', 'cabin', 'ad, advertisement, advertizement, advertising, advertizing, advert', 'amphitheater, amphitheatre, coliseum', 'monument', 'henhouse', 'cockpit', 'heater, warmer', 'windmill, aerogenerator, wind generator', 'pool', 'elevator, lift', 'decoration, ornament, ornamentation', 'labyrinth', 'text, textual matter', 'printer', 'mezzanine, first balcony', 'mattress', 'straw', 'stalls', 'patio, terrace', 'billboard, hoarding', 'bus stop', 'trouser, pant', 'console table, console', 'rack', 'notebook', 'shrine', 'pantry', 'cart', 'steam shovel', 'porch', 'postbox, mailbox, letter box', 'figurine, statuette', 'recycling bin', 'folding screen', 'telescope', 'deck chair, beach chair', 'kennel', 'coffee maker', "altar, communion table, lord's table", 'fish', 'easel', 'artificial golf green', 'iceberg', 'candlestick, candle holder', 'shower stall, shower bath', 'television stand', 'wall socket, wall plug, electric outlet, electrical outlet, outlet, electric receptacle', 'skeleton', 'grand piano, grand', 'candy, confect', 'grille door', 'pedestal, plinth, footstall', 'jersey, t-shirt, tee shirt', 'shoe', 'gravestone, headstone, tombstone', 'shanty', 'structure', 'rocking chair, rocker', 'bird', 'place mat', 'tomb', 'big top', 'gas pump, gasoline pump, petrol pump, island dispenser', 'lockers', 'cage', 'finger', 'bleachers', 'ferris wheel', 'hairdresser chair', 'mat', 'stands', 'aquarium, fish tank, marine museum', 'streetcar, tram, tramcar, trolley, trolley car', 'napkin, table napkin, serviette', 'dummy', 'booklet, brochure, folder, leaflet, pamphlet', 'sand trap', 'shop, store', 'table cloth', 'service station', 'coffin', 'drawer', 'cages', 'slot machine, coin machine', 'balcony', 'volleyball court', 'table tennis', 'control table', 'shirt', 'merchandise, ware, product', 'railway', 'parterre', 'chimney', 'can, tin, tin can', 'tanks', 'fabric, cloth, material, textile', 'alga, algae', 'system', 'map', 'greenhouse', 'mug', 'barbecue', 'trailer', 'toilet tissue, toilet paper, bathroom tissue', 'organ', 'dishrag, dishcloth', 'island', 'keyboard', 'trench', 'basket, basketball hoop, hoop', 'steering wheel, wheel', 'pitcher, ewer', 'goal', 'bread, breadstuff, staff of life', 'beds', 'wood', 'file cabinet', 'newspaper, paper', 'motorboat', 'rope', 'guitar', 'rubble', 'scarf', 'barrels', 'cap', 'leaves', 'control tower', 'dashboard', 'bandstand', 'lectern', 'switch, electric switch, electrical switch', 'baseboard, mopboard, skirting board', 'shower room', 'smoke', 'faucet, spigot', 'bulldozer', 'saucepan', 'shops', 'meter', 'crevasse', 'gear', 'candelabrum, candelabra', 'sofa bed', 'tunnel', 'pallet', 'wire, conducting wire', 'kettle, boiler', 'bidet', 'baby buggy, baby carriage, carriage, perambulator, pram, stroller, go-cart, pushchair, pusher', 'music stand', 'pipe, tube', 'cup', 'parking meter', 'ice hockey rink', 'shelter', 'weeds', 'temple', 'patty, cake', 'ski slope', 'panel', 'wallet', 'wheel', 'towel rack, towel horse', 'roundabout', 'canister, cannister, tin', 'rod', 'soap dispenser', 'bell', 'canvas', 'box office, ticket office, ticket booth', 'teacup', 'trellis', 'workbench', 'valley, vale', 'toaster', 'knife', 'podium', 'ramp', 'tumble dryer', 'fireplug, fire hydrant, plug', 'gym shoe, sneaker, tennis shoe', 'lab bench', 'equipment', 'rocky formation', 'plastic', 'calendar', 'caravan', 'check-in-desk', 'ticket counter', 'brush', 'mill', 'covered bridge', 'bowling alley', 'hanger', 'excavator', 'trestle', 'revolving door', 'blast furnace', 'scale, weighing machine', 'projector', 'soap', 'locker', 'tractor', 'stretcher', 'frame', 'grating', 'alembic', 'candle, taper, wax light', 'barrier', 'cardboard', 'cave', 'puddle', 'tarp', 'price tag', 'watchtower', 'meters', 'light bulb, lightbulb, bulb, incandescent lamp, electric light, electric-light bulb', 'tracks', 'hair dryer', 'skirt', 'viaduct', 'paper towel', 'coat', 'sheet', 'fire extinguisher, extinguisher, asphyxiator', 'water wheel', 'pottery, clayware', 'magazine rack', 'teapot', 'microphone, mike', 'support', 'forklift', 'canyon', 'cash register, register', 'leaf, leafage, foliage', 'remote control, remote', 'soap dish', 'windshield, windscreen', 'cat', 'cue, cue stick, pool cue, pool stick', 'vent, venthole, vent-hole, blowhole', 'videos', 'shovel', 'eaves', 'antenna, aerial, transmitting aerial', 'shipyard', 'hen, biddy', 'traffic cone', 'washing machines', 'truck crane', 'cds', 'niche', 'scoreboard', 'briefcase', 'boot', 'sweater, jumper', 'hay', 'pack', 'bottle rack', 'glacier', 'pergola', 'building materials', 'television camera', 'first floor', 'rifle', 'tennis table', 'stadium', 'safety belt', 'cover', 'dish rack', 'synthesizer', 'pumpkin', 'gutter', 'fruit stand', 'ice floe, floe', 'handle, grip, handgrip, hold', 'wheelchair', 'mousepad, mouse mat', 'diploma', 'fairground ride', 'radio', 'hotplate', 'junk', 'wheelbarrow', 'stream', 'toll plaza', 'punching bag', 'trough', 'throne', 'chair desk', 'weighbridge', 'extractor fan', 'hanging clothes', 'dish, dish aerial, dish antenna, saucer', 'alarm clock, alarm', 'ski lift', 'chain', 'garage', 'mechanical shovel', 'wine rack', 'tramway', 'treadmill', 'menu', 'block', 'well', 'witness stand', 'branch', 'duck', 'casserole', 'frying pan', 'desk organizer', 'mast', 'spectacles, specs, eyeglasses, glasses', 'service elevator', 'dollhouse', 'hammock', 'clothes hanging', 'photocopier', 'notepad', 'golf cart', 'footpath', 'cross', 'baptismal font', 'boiler', 'skip', 'rotisserie', 'tables', 'water mill', 'helmet', 'cover curtain', 'brick', 'table runner', 'ashtray', 'street box', 'stick', 'hangers', 'cells', 'urinal', 'centerpiece', 'portable fridge', 'dvds', 'golf club', 'skirting board', 'water cooler', 'clipboard', 'camera, photographic camera', 'pigeonhole', 'chips', 'food processor', 'post box', 'lid', 'drum', 'blender', 'cave entrance', 'dental chair', 'obelisk', 'canoe', 'mobile', 'monitors', 'pool ball', 'cue rack', 'baggage carts', 'shore', 'fork', 'paper filer', 'bicycle rack', 'coat rack', 'garland', 'sports bag', 'fish tank', 'towel dispenser', 'carriage', 'brochure', 'plaque', 'stringer', 'iron', 'spoon', 'flag pole', 'toilet brush', 'book stand', 'water faucet, water tap, tap, hydrant', 'ticket office', 'broom', 'dvd', 'ice bucket', 'carapace, shell, cuticle, shield', 'tureen', 'folders', 'chess', 'root', 'sewing machine', 'model', 'pen', 'violin', 'sweatshirt', 'recycling materials', 'mitten', 'chopping board, cutting board', 'mask', 'log', 'mouse, computer mouse', 'grill', 'hole', 'target', 'trash bag', 'chalk', 'sticks', 'balloon', 'score', 'hair spray', 'roll', 'runner', 'engine', 'inflatable glove', 'games', 'pallets', 'baskets', 'coop', 'dvd player', 'rocking horse', 'buckets', 'bread rolls', 'shawl', 'watering can', 'spotlights', 'post-it', 'bowls', 'security camera', 'runner cloth', 'lock', 'alarm, warning device, alarm system', 'side', 'roulette', 'bone', 'cutlery', 'pool balls', 'wheels', 'spice rack', 'plant pots', 'towel ring', 'bread box', 'video', 'funfair', 'breads', 'tripod', 'ironing board', 'skimmer', 'hollow', 'scratching post', 'tricycle', 'file box', 'mountain pass', 'tombstones', 'cooker', 'card game, cards', 'golf bag', 'towel paper', 'chaise lounge', 'sun', 'toilet paper holder', 'rake', 'key', 'umbrella stand', 'dartboard', 'transformer', 'fireplace utensils', 'sweatshirts', 'cellular telephone, cellular phone, cellphone, cell, mobile phone', 'tallboy', 'stapler', 'sauna', 'test tube', 'palette', 'shopping carts', 'tools', 'push button, push, button', 'star', 'roof rack', 'barbed wire', 'spray', 'ear', 'sponge', 'racket', 'tins', 'eyeglasses', 'file', 'scarfs', 'sugar bowl', 'flip flop', 'headstones', 'laptop bag', 'leash', 'climbing frame', 'suit hanger', 'floor spotlight', 'plate rack', 'sewer', 'hard drive', 'sprinkler', 'tools box', 'necklace', 'bulbs', 'steel industry', 'club', 'jack', 'door bars', 'control panel, instrument panel, control board, board, panel', 'hairbrush', 'napkin holder', 'office', 'smoke detector', 'utensils', 'apron', 'scissors', 'terminal', 'grinder', 'entry phone', 'newspaper stand', 'pepper shaker', 'onions', 'central processing unit, cpu, c p u , central processor, processor, mainframe', 'tape', 'bat', 'coaster', 'calculator', 'potatoes', 'luggage rack', 'salt', 'street number', 'viewpoint', 'sword', 'cd', 'rowing machine', 'plug', 'andiron, firedog, dog, dog-iron', 'pepper', 'tongs', 'bonfire', 'dog dish', 'belt', 'dumbbells', 'videocassette recorder, vcr', 'hook', 'envelopes', 'shower faucet', 'watch', 'padlock', 'swimming pool ladder', 'spanners', 'gravy boat', 'notice board', 'trash bags', 'fire alarm', 'ladle', 'stethoscope', 'rocket', 'funnel', 'bowling pins', 'valve', 'thermometer', 'cups', 'spice jar', 'night light', 'soaps', 'games table', 'slotted spoon', 'reel', 'scourer', 'sleeping robe', 'desk mat', 'dumbbell', 'hammer', 'tie', 'typewriter', 'shaker', 'cheese dish', 'sea star', 'racquet', 'butane gas cylinder', 'paper weight', 'shaving brush', 'sunglasses', 'gear shift', 'towel rail', 'adding machine, totalizer, totaliser')
    # 847

    def __init__(self, **kwargs):
        super(MyADE20KDataset847, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.tif',
            reduce_zero_label=False,
            **kwargs)

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            # The  index range of official requirement is from 0 to 150.
            # But the index range of output is from 0 to 149.
            # That is because we set reduce_zero_label=True.
            result = result + 1

            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for ade20k evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
               the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)
        return result_files