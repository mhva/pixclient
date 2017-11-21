# -*- coding: utf-8 -*-

pixclient_config = {
  # Login credentials.
  'username': ‘username’,
  'password': ‘password’,

  # Proxy server URL (cURL syntax).
  #’proxy': 'socks5://127.0.0.1:8080',

  # User agent.
  'user-agent': 'Mozilla/5.0 (Windows NT 6.1; rv:45.0) Gecko/20100101 Firefox/52.0’,

  'max-retries': 3,
  'backoff-interval': 5,
  'backoff-exponent': 2.5,
  'backoff-limit': 60
}
