#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import socket
import requests

import argparse
import errno
import json
import math
import os
import pprint
import re
import socket
import subprocess
import sys
import time
import urlparse

from contextlib import closing
from itertools import *
from urllib import quote_plus
from stat import S_ISDIR

from pysocks import socks
from pixivpy3.papi import PixivAPI
from pixivpy3.utils import PixivError

class ServiceError(Exception):
  """
  Represents a basic service error, an error that's returned by pixiv when
  interacting with its APIs.
  """
  def __init__(self):
    super(ServiceError, self).__init__()

class UnknownServiceError(ServiceError):
  """
  A generic service error that does not have an associated error message.
  """
  def __init__(self, response):
    super(UnknownServiceError, self).__init__()
    self.response = response

class OtherServiceError(ServiceError):
  """
  An unclassified, yet a well-defined error (compared to
  the UnknownServiceError). Error message that was returned by pixiv can be
  read from the |error_text| member variable.
  """
  def __init__(self, text):
    super(OtherServiceError, self).__init__()
    self.error_text = text

class AuthenticationError(ServiceError):
  def __init__(self, text):
    super(AuthenticationError, self).__init__()
    self.error_text = text

class IllustrationDoesNotExistError(ServiceError):
  def __init__(self):
    super(IllustrationDoesNotExistError, self).__init__()

class ProxyConnectionFailure(Exception):
  def __init__(self, proxy_url):
    self.proxy_url = proxy_url

class DownloadFailure(Exception):
  def __init__(self, url, local_file, error):
    super(DownloadFailure, self).__init__()
    self.remote_url = url
    self.local_file = local_file
    self.error = error

class FileAlreadyExistsError(Exception):
  def __init__(self, local_file):
    super(FileAlreadyExistsError, self).__init__()
    self.local_file = local_file

class UnsupportedImageQualityError(Exception):
  def __init__(self, image_dict):
    super(UnsupportedImageQualityError, self).__init__()
    self.bad_image_dict = image_dict

class IllustMetadata(object):
  def __init__(self, metadata):
    self._metadata = metadata

  def get_id(self):
    """Returns a numeric id of the artwork."""
    return int(self._metadata['id'])

  def get_title(self):
    """Returns title of the artwork."""
    return self._metadata['title']

  def get_description(self):
    """Returns a description of the artwork."""
    return self._metadata.get('caption', '')

  def get_upload_date(self):
    """Returns upload date as a string."""
    return self._metadata.get('reuploaded_time')

  def get_author_id(self):
    """Returns author id."""
    return int(self._metadata.get('user', {}).get('id', -1))

  def get_author_name(self):
    """Returns author display name."""
    return self._metadata.get('user', {}).get('name')

  def get_author_account(self):
    """Returns author account name."""
    return self._metadata.get('user', {}).get('account')

  def get_tags(self):
    """Returns a list of tags."""
    return self._metadata.get('tags', [])

  def is_multipage(self):
    """Returns True, if the the artwork contains multiple images."""
    m = self._metadata.get('metadata', None)
    return (m.get('pages', None) is not None) if m is not None else False

  def is_manga(self):
    """Returns True, if the artwork is manga."""
    # XXX: Should we query self._metadata['is_manga'] instead?
    return self.is_multipage()

  def get_image_urls(self):
    """Returns a list of images associated with this pixiv artwork."""
    if self.is_multipage():
      pages = self._metadata['metadata']['pages']
      image_urls = map(
        lambda x: IllustMetadata._choose_best_image(x['image_urls']), pages)
      if None not in image_urls:
        return image_urls
      else:
        raise UnsupportedImageQualityError(pages[image_urls.index(None)])
    else:
      url = IllustMetadata._choose_best_image(self._metadata['image_urls'])
      if url is not None:
        return [url]
      else:
        raise UnsupportedImageQualityError(self._metadata['image_urls'])

  @staticmethod
  def _choose_best_image(image_dict):
    """
    Returns URL of the image with the best quality.
    Returns None, if no URL with a known image quality was found.
    """
    for p in ['large', 'medium']:
      url = image_dict.get(p)
      if url is not None:
        return url
    return None

class PixclientPixivAPI(object):
  def __init__(self, username, password, auth_cache_file=None):
    super(PixclientPixivAPI, self).__init__()
    self._api = None
    self._username = username
    self._password = password
    if auth_cache_file is not None and len(auth_cache_file.strip()) > 0:
      self._auth_cache_file = auth_cache_file
    else:
      self._auth_cache_file = None

  def _make_pixiv_api(self):
    pixiv_api = PixivAPI()
    pixiv_api.set_accept_language('ja-JP')
    return pixiv_api

  def _login_using_password(self):
    """
    Authenticates with pixiv using login/password pair. May throw
    an AuthenticationError exception, if given login credentials are incorrect.

    Returns a PixivAPI object.
    """
    pixiv_api = self._make_pixiv_api()

    try:
      pixiv_api.login(self._username, self._password)
    except PixivError as e:
      raise AuthenticationError('Username or password is incorrect')
    return pixiv_api

  def _login_using_cached_session_token(self):
    """
    Builds a PixivAPI object using cached session token.
    Returns a PixivAPI object or None, if there's no cached token.
    """
    if self._auth_cache_file is not None:
      try:
        with open(self._auth_cache_file, 'r') as f:
          auth_info = json.load(f)

        if auth_info.get('session_token') is not None \
            and auth_info.get('refresh_token') is not None:
          pixiv_api = self._make_pixiv_api()
          pixiv_api.set_auth(
            auth_info['session_token'],
            auth_info['refresh_token'])
          return pixiv_api
      except IOError as e:
        if e.errno != errno.ENOENT:
          raise
      except PixivError:
        raise
      except Exception as e:
        pass
    return None

  def _login_using_refresh_token(self):
    raise RuntimeError("__login_using_refresh_token: not implemented")

  def _login(self, force_relogin=False):
    """
    Tries to authenticate with pixiv using username/password pair provided
    in constructor. If session token has already been cached, does nothing.

    Returns a PixivAPI object.
    """
    pixiv_api = None

    if not force_relogin and self._api is not None:
      return
    if not force_relogin:
      pixiv_api = self._login_using_cached_session_token()

    if pixiv_api is None:
      pixiv_api = self._login_using_password()

      # Save session token to cache.
      if self._auth_cache_file is not None:
        with open(self._auth_cache_file, 'w') as f:
          auth_info = {
            'session_token': pixiv_api.access_token,
            'refresh_token': pixiv_api.refresh_token
          }
          json.dump(auth_info, f)

    self._api = pixiv_api

  def __authenticated_call(f):
    """
    Function decorator that will add auto-login capability to authenticated
    API calls. If it detects an authentication error in server's response,
    it will attempt to refresh session token and will invoke target function
    one more time.
    """
    def wrap(self, *args, **kwargs):
      self._login()
      response = f(self, *args, **kwargs)
      try:
        if response.get('status', 'failure') == 'failure':
          msg = response['errors']['system']['message']
          if msg.find('access token provided is invalid') != -1:
            self._login(force_relogin=True)
            return f(self, *args, **kwargs)
          return response
      except KeyError:
        pass
      return response
    return wrap

  @__authenticated_call
  def _api_call_works(self, illust_id):
    return self._api.works(illust_id)

  def artwork_info(self, illust_id):
    """
    Returns metadata for pixiv illustration with |illust_id| id.
    Returns a list of IllustMetadata objects.
    """
    response = self._api_call_works(illust_id)

    if response.get('status', 'failure') == 'success':
      # It seems that server response could have multiple metadata blobs (for
      # different illustrations?). I haven't encountered the case with the multiple
      # blobs out in the wild yet, but to be on the safe side treat each blob as a
      # separate artwork.
      illust_descriptors = response['response']
      return [IllustMetadata(x) for x in illust_descriptors]
    else:
      try:
        if 'code' in response['errors']['system'] \
            and response['errors']['system']['code'] == 206:
          raise IllustrationDoesNotExistError()
        raise OtherServiceError(response['errors']['system']['message'])
      except IllustrationDoesNotExistError:
        raise
      except OtherServiceError:
        raise
      except:
        raise UnknownServiceError(response)

def print_stderr(s):
  sys.stderr.write('%s\n' % s)
  sys.stderr.flush()

def print_info(s):
  print_stderr('\033[94m%s\033[0m' % s)

def print_debug(s):
  print_stderr('%s' % s)

def print_error(s):
  print_stderr('\033[91;1m%s\033[0m' % s)

def die(s, exit_code=1):
  print_error(s)
  exit(exit_code)

def dump_response(response):
  print_debug(pprint.pformat(response, indent=2))

def setup_proxy(url):
  scheme_map = {
    'socks4': socks.SOCKS4, 'socks4h': socks.SOCKS4,
    'socks5': socks.SOCKS5, 'socks5h': socks.SOCKS5
  }
  rdns_map = {
    'socks4': False, 'socks4h': True,
    'socks5': False, 'socks5h': True
  }
  name_map = {
    'socks4': 'SOCKS v4', 'socks4h': 'SOCKS v4 (w/ Remote DNS)',
    'socks5': 'SOCKS v5', 'socks5h': 'SOCKS v5 (w/ Remote DNS)'
  }

  url_object = urlparse.urlparse(url)
  proxy_type = scheme_map[url_object.scheme]
  address    = url_object.hostname
  port       = url_object.port
  rdns       = rdns_map[url_object.scheme]
  user       = url_object.username
  password   = url_object.password

  print_info('Using %s proxy at %s port %d' % \
    (name_map[url_object.scheme], address, port))

  connectable_address = None
  connectable_port = None
  gai_addrs = socket.getaddrinfo(address, port, 0, 0, socket.IPPROTO_TCP)

  for addr in gai_addrs:
    with closing(socket.socket(addr[0], addr[1], socket.IPPROTO_TCP)) as sock:
      try:
        sock.connect((addr[4][0], addr[4][1]))
        (connectable_address, connectable_port) = addr[4]
      except:
        continue

  if connectable_address is None or connectable_port is None:
    raise ProxyConnectionFailure(url)

  socks.set_default_proxy(proxy_type, address, port, rdns, user, password)
  socket.socket = socks.socksocket

def guess_file_extension(url):
  default_ext = u'png'
  url_object = urlparse.urlparse(url)
  fileext = u''.join(reversed(list(takewhile(lambda c: c != u'.',
                                       reversed(url_object.path)))))
  # Sanitize resulting extension. If it fails return default extension.
  # Most image software doesn't care about file extension anyway.
  if fileext.isalnum() and len(fileext) < len(url_object.path):
    return fileext
  else:
    return default_ext

def pixclient_path():
  return os.path.join(sys.path[0], './')

def utf8_please(o):
  """
  Undoes unicode conversion of strings that json module performs by default.
  Returns a new dictionary containing utf-8 encoded str's instead of
  unicode objects.
  """
  if isinstance(o, list):
    return map(lambda x: utf8_please(x), o)
  elif isinstance(o, dict):
    return {utf8_please(k): utf8_please(v) for k, v in o.items() }
  elif isinstance(o, unicode):
    return o.encode('utf-8')
  else:
    return o

def load_config():
  """
  Reads config from file and returns a dictionary with config entries.
  """

  # HACK: Find location of config file from args. This is needed so we can
  # access config data early, even before parsing other arguments.
  bootstrap_argp = argparse.ArgumentParser(add_help=False)
  bootstrap_argp.add_argument('-c', '--config', dest='config')
  bootstrap_args = bootstrap_argp.parse_known_args(sys.argv)[0]

  if bootstrap_args.config is not None:
    filename = bootstrap_args.config
  else:
    return {}

  try:
    with open(filename, 'r') as f:
      # Remove comments.
      pseudo_json = f.read()
      clean_json = re.sub(r'^[^\S\r\n]*\/\/.*$', '', pseudo_json, flags=re.MULTILINE)

      return process_config(utf8_please(json.loads(clean_json)))
  except ValueError as e:
    die("Malformed JSON content in config file. (%s)" % str(e))
  except IOError as e:
    die("IO error while reading config file. (%s)" % e.strerror)
  except Exception as e:
    die("Unable to process config file. (%s)" % str(e))

def process_config(config):
  # Expand path in auth_cache_file. If it's relative, convert to absolute.
  if 'auth_cache_file' in config:
    auth_cache_file = config['auth_cache_file']
    if len(auth_cache_file) > 0:
      auth_cache_file = os.path.expandvars(auth_cache_file)
      if not os.path.isabs(auth_cache_file):
        auth_cache_file = os.path.join(pixclient_path(), auth_cache_file)
    else:
      auth_cache_file = None
    config['auth_cache_file'] = auth_cache_file
  return config

def make_directory(name, mode=0755):
  """
  Creates a new directory with name `name` and file permissions `mode`.

  If the directory already exists, does nothing.
  Throws `OSError`, if anything goes wrong.
  """
  path = os.path.abspath(name)
  try:
    print_info('Creating directory %s (mode 0%o)' % (name, mode))
    os.mkdir(path, mode)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise
    else:
      print_debug('Directory %s already exists' % name)

class BackoffStrategy(object):
  def __init__(self, backoff_interval=None, backoff_exponent=None,
               backoff_limit=None, max_retries=None):
    self._backoff_interval = backoff_interval or 5
    self._backoff_exponent = backoff_exponent or 2.5
    self._backoff_limit    = backoff_limit or 60
    self._max_retries      = max_retries or 3
    self.reset()

  def reset(self):
    self._retries_remain = self._max_retries
    self._current_backoff_interval = self._backoff_interval

  def can_back_off(self):
    return self._retries_remain > 0

  def back_off(self):
    if self.can_back_off():
      time.sleep(self._current_backoff_interval)

      interval = int(self._current_backoff_interval * self._backoff_exponent)
      self._current_backoff_interval = min(self._backoff_limit, interval)
      self._retries_remain -= 1
      return True
    else:
      return False

class DownloadProgress(object):
  """Draws a simplistic download progress bar in terminal."""

  def _convert_to_units(self, byte_value):
    gb = 1 * 1024 * 1024 * 1024
    mb = 1 * 1024 * 1024
    kb = 1 * 1024

    if byte_value / gb > 0:
      return ('%.2f' % (float(byte_value) / gb), 'G')
    elif byte_value / mb > 0:
      return ('%.2f' % (float(byte_value) / mb), 'M')
    elif byte_value / kb > 0:
      return (byte_value / kb, 'K')
    else:
      return (str(byte_value), '')

  def _draw(self, text):
    try:
      sys.stderr.write(text)
    except:
      pass

  def _clear(self):
    self._draw('\r')

  def _draw_percentage(self, done=None, total=None):
    percentage_length = len('%.2f%%' % 100.0)

    if done and total:
      percentage = '%.2f%%' % (float(done) / total * 100)
      space_filler = ' ' * (percentage_length - len(percentage))
      self._draw(space_filler + percentage + ' ')
    else:
      self._draw('--.--% ')

  def _draw_progressbar(self, done=None, total=None):
    progress_length = 25
    if done and total:
      progress_used = int(math.ceil(progress_length * (float(done) / total)))
      progress_empty = progress_length - progress_used
      progress_bar = '[%s%s]' % ('#' * progress_used, ' ' * progress_empty)
      self._draw(progress_bar)
    else:
      self._draw('[' + ' ' * progress_length + ']')

  def _draw_numeric_progress(self, done=None, total=None):
    numeric_progress_length = 14

    if done and total:
      done_units = self._convert_to_units(done)
      total_units = self._convert_to_units(total)
      numeric_progress = '(%s%s / %s%s)' % \
          (done_units[0], done_units[1], total_units[0], total_units[1])
    elif done:
      done_units = self._convert_to_units(done)
      numeric_progress = '(%s%s / --)' % (done_units[0], done_units[1])
    else:
      numeric_progress = '(-- / --)'

    space_filler = ' ' * (numeric_progress_length - len(numeric_progress))
    self._draw(space_filler + numeric_progress + ' ')

  def _draw_title(self, title):
    self._draw(title)

  def draw_preheating_phase(self, title):
    """
    Displays an empty progress in terminal. This allows for basic feedback
    to be displayed when the program is waiting for data (i.e. waiting
    while a TCPconnection is being established).
    """
    self._clear()
    self._draw_percentage()
    self._draw_progressbar()
    self._draw_numeric_progress()
    self._draw_title(title)

  def draw_progress(self, title, done, total=None):
    """
    Displays a single-line progress bar in terminal.
    """
    self._clear()
    self._draw_percentage(done, total)
    self._draw_progressbar(done, total)
    self._draw_numeric_progress(done, total)
    self._draw_title(title)

  def finalize(self):
    """
    Inserts a linefeed. Use this when a task is finished or no further
    progress is going to be made.
    """
    self._draw('\n')

class CURLDownloadSession(object):
  def __init__(self, backoff_strategy=None, proxy_url=None, user_agent=None):
    """
    Creates new CURLDownloadSession object.
    @param str proxy_url   string containing socks proxy URL
    @param int max_retries max download retries before giving up
    """
    self._backoff_strat = backoff_strategy or BackoffStrategy()
    self._proxy_url = proxy_url
    self._curl_base_command = [
        'curl'
      , '-L' # Follow redirects.
      , '-f' # Do not save error documents, enables error 22.
      ## Pixiv blocks access to a range of images (only NSFW?) if not given
      ## a correct PHPSESSID (user account may need to allow displaying NSFW
      ## images).
      #, '-b', 'PHPSESSID=%s' % urllib.quote_plus(session_id)
    ]
    if proxy_url:
      self._curl_base_command.extend(['-x', proxy_url])
    if user_agent:
      self._curl_base_command.extend(['-A', user_agent])

  def download(self, url, out_file, referrer=None, cookies={}):
    """
    Saves a document located at @url to a file named @out_file.
    @param unicode url      document URL
    @param unicode output   output directory
    @param unicode referrer value for the Referer header
    @param dict    cookies  `cookie_name -> cookie_value` dictionary
    """
    self._backoff_strat.reset()

    cookie_args = []
    for k, v in cookies.items():
      cookie_args.append('-b')
      cookie_args.append('%s=%s' % (k, quote_plus(v)))

    curl_command = self._curl_base_command \
        + ['-o', out_file] \
        + (['-e', referrer + ';auto'] if referrer else []) \
        + cookie_args \
        + [url]

    while True:
      result = subprocess.call(curl_command)
      if result == 0:
        break
      elif result == 18:
        # Retry for free, if cURL returned partial file error (18).
        if '-C' not in curl_command:
          curl_command = ['-C', '-'] + curl_command
      elif self._backoff_strat.can_back_off():
        self._backoff_strat.back_off()
      else:
        raise DownloadFailure(url, out_file, 'cURL error (%d)' % result)

class PyRequestsDownloadSession(object):
  def __init__(self, backoff_strategy=None, user_agent=None, keep_alive=True):
    """
    Creates new PyRequestsDownloadSession object.
    Use SocksiPy for SOCKS proxy support.
    """
    self._backoff_strat = backoff_strategy or BackoffStrategy()
    self._session = requests.Session()
    self._session.keep_alive = keep_alive
    if user_agent:
      self._session.headers['User-Agent'] = user_agent

  def _retry_download(self, url, out_file,
                      ui_title=None, progress_ui=None,
                      referrer=None, cookies={}):
    response = self._session.get(
        url
      , headers={'Referer': referrer} if referrer else {}
      , cookies=cookies
      , allow_redirects=True
      , stream=True
    )
    response.raise_for_status()

    with open(out_file, 'wb') as f:
      try:
        datasz = long(response.headers['Content-Length'])
      except:
        datasz = None
      if datasz and datasz > 0:
        progress_fn = lambda x: progress_ui.draw_progress(ui_title, x, datasz)
      else:
        progress_fn = lambda x: progress_ui.draw_progress(ui_title, x)

      # Save response's body to file while periodically updating progress.
      written_out = 0
      progress_update_time = 0.0
      for data in response.iter_content(chunk_size=32 * 1024):
        f.write(data)

        current_time = time.time()
        written_out += len(data)
        if current_time - progress_update_time > 0.1:
          progress_fn(written_out)
          progress_update_time = current_time

      # Update progress one last time to reflect the successful download.
      progress_fn(written_out)

  def download(self, url, out_file, referrer=None, cookies={}):
    """
    Saves a document located at @url to a file named @out_file.
    @param unicode url      document URL
    @param unicode output   output directory
    @param unicode referrer value for the Referer header
    @param dict    cookies  `cookie_name -> cookie_value` dictionary
    """
    ui_title    = os.path.basename(out_file)
    progress_ui = DownloadProgress()
    progress_ui.draw_preheating_phase(ui_title)

    self._backoff_strat.reset()

    while True:
      try:
        self._retry_download(
            url
          , out_file
          , ui_title=ui_title
          , progress_ui=progress_ui
          , referrer=referrer
          , cookies=cookies
        )
        break
      except ( requests.exceptions.RequestException
             , requests.exceptions.ConnectionError
             , requests.exceptions.TooManyRedirects
             , requests.exceptions.Timeout
             , requests.exceptions.HTTPError
             ) as e:
        if not self._backoff_strat.back_off():
          raise DownloadFailure(url, out_file, e)
      except IOError as e:
        raise DownloadFailure(url, out_file, e)
      finally:
        progress_ui.finalize()

def fetch_artwork(illust_metadata, download_session,
                  output=u'.', delay=0, keep_going=False):
  """
  Downloads all images associated with given artwork.

  @param IllustMetadata illust_metadata artwork metadata
  @param unicode output     output directory
  @param int     delay      delay (in seconds) between each download
  @param bool    keep_going don't stop on error
  """
  image_urls = illust_metadata.get_image_urls()
  dest_dir = os.path.join(output, u'%d' % illust_metadata.get_id())
  if illust_metadata.is_manga():
    referrer = (u'https://www.pixiv.net/member_illust.php?mode=manga&' + \
                u'illust_id=%d') % illust_metadata.get_id()
  else:
    referrer = (u'https://www.pixiv.net/member_illust.php?mode=medium&' + \
                u'illust_id=%d') % illust_metadata.get_id()
  failed_downloads = []

  make_directory(dest_dir)

  # Download all files in the artwork.
  for (i, url) in zip(xrange(len(image_urls)), image_urls):
    filename = '%d_p%d.%s' % \
      (illust_metadata.get_id(), i, guess_file_extension(url))
    filename = os.path.join(dest_dir, filename)
    print_info('(%.3d/%.3d) Saving \'%s\' as %s' % \
      (i + 1, len(image_urls), url, filename))

    if os.path.exists(filename) and not keep_going:
      raise FileAlreadyExistsError(filename)

    try:
      download_session.download(url, filename, referrer=referrer)
    except DownloadFailure as e:
      if keep_going:
        print_error('Ignoring download error (keep going): %s', url)
        failed_downloads.append(url)
      else:
        raise e

    if delay > 0 and i < len(image_urls) - 1:
      time.sleep(delay)

  # Write title and description in a separate file.
  description_filename = 'description.txt'
  print_info('Writing description to %s' % description_filename)

  with open(os.path.join(dest_dir, description_filename), 'w') as f:
    author_display_name = illust_metadata.get_author_name()
    author_id = illust_metadata.get_author_id()
    author_account = illust_metadata.get_author_account()

    posting_date = illust_metadata.get_upload_date()
    tags = ', '.join(illust_metadata.get_tags())

    title = illust_metadata.get_title()
    title = title.strip() if title else None

    description = illust_metadata.get_description()
    description = description.rstrip() if description else None

    f.write(ur"""
AUTHR: {0} ({1}:{2})
DATE : {3}
TAGS : {4}

TITLE: {5}
DESCR: {6}
      """.strip().format(
        author_display_name,
        author_id,
        author_account,
        posting_date,
        tags,
        title,
        description).encode('utf-8')
    )

  print_info('Files were saved to: %s' % dest_dir)
  return failed_downloads

def subcommand_illust():
  def unicode_arg(x):
    return x.decode(sys.getfilesystemencoding())

  pixclient_config = load_config()

  credentials_required = (pixclient_config.get('username') is None) \
    or (pixclient_config.get('password') is None)

  argp = argparse.ArgumentParser(prog='%s illust' % sys.argv[0],
    description='fetch a single image or a complete collection from Pixiv')
  argp.add_argument('-u', '--user', dest='user',
    default=pixclient_config.get('username'), help='pixiv login',
    required=credentials_required)
  argp.add_argument('-p', '--password', dest='password', metavar='PW',
    default=pixclient_config.get('password'), help='pixiv password',
    required=credentials_required)
  argp.add_argument('-c', '--config', dest='config', metavar='CFG',
    help='path to config file')
  argp.add_argument('--method', metavar='METHOD', dest='method',
    choices=['curl', 'requests'], help='download method',
    default=pixclient_config.get('download-method', 'requests'))
  argp.add_argument('--delay', metavar='DELAY', dest='delay', type=int,
    default=1, help='number of seconds to wait between downloads')
  argp.add_argument('--keep-going', dest='keep_going', action='store_true',
    default=False, help='keep going, even if some file has failed to download')
  argp.add_argument('--user-agent', dest='user_agent', metavar='UA',
    default=pixclient_config.get('user-agent'),
    help='user agent for HTTP requests')
  argp.add_argument('-o', '--output', metavar='DIR', dest='output',
    type=unicode_arg, default=u'.', help='output directory')
  # TODO: add better documentation.
  argp.add_argument('-x', '--proxy', metavar='URL', dest='proxy_url',
    default=pixclient_config.get('proxy'),
    help='SOCKS server URL (syntax is the same as cURL)')
  argp.add_argument('illust_id', type=int, metavar='illust-id',
    help='pixiv illustration id')
  args = argp.parse_args(sys.argv[2:])

  # Make sure output directory exists.
  try:
    st = os.stat(args.output)
    if not S_ISDIR(st.st_mode):
      die('Argument for (-o | --output) must be a directory name.')
  except OSError as e:
    if e.errno == errno.EEXIST:
      die('Output directory \'%s\' does not exist' % args.output)
    else:
      raise

  try:
    if args.proxy_url and len(args.proxy_url.strip()) > 0:
      setup_proxy(args.proxy_url)

    if args.user and args.password:
      print_info('Will log in as %s' % args.user)
    else:
      die("No valid login credentials found; login credentials must be "
          "specified using either --user/--password arguments or in config "
          "file")

    print_info('Fetching artwork metadata')
    pixiv_api = PixclientPixivAPI(args.user, args.password,
      auth_cache_file=pixclient_config.get('auth_cache_file'))
    artworks = pixiv_api.artwork_info(args.illust_id)

    backoff_strat = BackoffStrategy(
        backoff_interval=pixclient_config.get('backoff-interval')
      , backoff_exponent=pixclient_config.get('backoff-exponent')
      , backoff_limit=pixclient_config.get('backoff-limit')
      , max_retries=pixclient_config.get('max-retries')
    )

    if args.method == 'curl':
      download_session = CURLDownloadSession(
          backoff_strategy=backoff_strat
        , proxy_url=args.proxy_url
        , user_agent=args.user_agent
      )
    else:
      download_session = PyRequestsDownloadSession(
          backoff_strategy=backoff_strat
        , user_agent=args.user_agent
        , keep_alive=False if args.delay > 4 else True
      )

    for a in artworks:
      fetch_artwork(a, download_session,
        output=args.output,
        delay=args.delay,
        keep_going=args.keep_going)
  except AuthenticationError as e:
    # Set exit code to 2 to allow scripts to detect expired session.
    die("Authentication error: %s." % e.error_text, exit_code=2)
  except IllustrationDoesNotExistError:
    die("Illustration with id '%d' does not exist." % args.illust_id)
  except OtherServiceError as e:
    die("Pixiv server returned an error: %s." % e.error_text)
  except UnknownServiceError as e:
    dump_response(e.response)
    die("Pixiv returned unknown response")
  except DownloadFailure as e:
    die("Could not save remote document " + \
        "(%s) to a local file (%s): %s." % \
        (e.remote_url, os.path.basename(e.local_file), str(e.error)))
  except FileAlreadyExistsError as e:
    die(("File '%s' already exists in output directory. Use --keep-going " + \
         "flag, if you want to overwrite it.") % os.path.basename(e.local_file))
  except ProxyConnectionFailure as e:
    die("Couldn't connect to proxy: %s" % e.proxy_url)

if __name__ == '__main__':
  argp = argparse.ArgumentParser(
    description='Description goes here')
  argp.add_argument('verb', choices=['illust'],
    help='supported program verbs')
  args = argp.parse_args(sys.argv[1:2])
  if args.verb == 'illust':
    subcommand_illust()
  else:
    print_error("Unknown verb \'%s\'" % args.verb)
    exit(1)
