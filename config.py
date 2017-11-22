# -*- coding: utf-8 -*-

pixclient_config = {
  # Login credentials.
  'username': '',
  'password': '',

  # Proxy server URL (cURL syntax).
  # Example: socks5://127.0.0.1:8080
  'proxy': '',

  # Path to a file where a pixiv authorization token will be stored
  # which we use to skip authorization on repeated pixclient invocations.
  #
  # - Relative paths are allowed, the file will be saved relative to pixclient
  #   script location.
  # - Environment variables are supported (in form $NAME or ${NAME}).
  # - If left empty, pixclient will not store any auth data.
  'auth_cache_file': '${HOME}/.pixclient_auth_cache.json',

  # User agent.
  'user-agent': 'Mozilla/5.0 (Windows NT 6.1; rv:52.0) Gecko/20100101 Firefox/52.0',

  'max-retries': 3,
  'backoff-interval': 5,
  'backoff-exponent': 2.5,
  'backoff-limit': 60
}
