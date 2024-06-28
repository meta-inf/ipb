import logging, logging.handlers, queue, os


def _get_extra_dict():
    envs = ['SLURM_JOB_ID', 'HOSTNAME']
    return {k:os.environ[k] for k in envs if k in os.environ}


class CustomHTTPHandler(logging.handlers.HTTPHandler):

    def __init__(self, remote_url):
        toks = remote_url.split('/')
        host, url = toks[0], '/' + '/'.join(toks[1:])
        super().__init__(host, url, method='POST', secure=False)
        self._extra_dict = _get_extra_dict()

    def mapLogRecord(self, record):
        return record.__dict__ | self._extra_dict


def get_logger(remote_url=None, log_level=logging.INFO):
    logger = logging.getLogger()
    logging.addLevelName(
        logging.INFO,
        "\033[1;34m%s\033[1;0m" % logging.getLevelName(logging.INFO))
    logging.addLevelName(
        logging.WARNING,
        "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
    logging.addLevelName(
        logging.ERROR,
        "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))

    if remote_url is not None and len(remote_url.strip()) > 0:
        http_handler = CustomHTTPHandler(remote_url)
        que = queue.Queue(-1)
        que_listener = logging.handlers.QueueListener(que, http_handler)
        que_handler = logging.handlers.QueueHandler(que)
        que_listener.start()
        logger.addHandler(que_handler)
    else:
        que_listener = None

    logger.setLevel(log_level)

    return logger, que_listener
