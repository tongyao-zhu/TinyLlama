

export ELASTIC_PASSWORD=YL8RhK0Ua*PXt_Fn1LFW
export ES_HOME=/home/aiops/zhuty/elasticsearch-8.12.1
curl_with_prefix="curl --cacert $ES_HOME/config/certs/http_ca.crt -u elastic:$ELASTIC_PASSWORD "
echo "es_home: $ES_HOME es_password: $ELASTIC_PASSWORD"

$curl_with_prefix https://localhost:9200
$curl_with_prefix -X POST "https://localhost:9200/_bulk?pretty" -H 'Content-Type: application/json' -d'
{ "index" : { "_index" : "books" } }
{"name": "Revelation Space", "author": "Alastair Reynolds", "release_date": "2000-03-15", "page_count": 585}
{ "index" : { "_index" : "books" } }
{"name": "1984", "author": "George Orwell", "release_date": "1985-06-01", "page_count": 328}
{ "index" : { "_index" : "books" } }
{"name": "Fahrenheit 451", "author": "Ray Bradbury", "release_date": "1953-10-15", "page_count": 227}
{ "index" : { "_index" : "books" } }
{"name": "Brave New World", "author": "Aldous Huxley", "release_date": "1932-06-01", "page_count": 268}
{ "index" : { "_index" : "books" } }
{"name": "The Handmaids Tale", "author": "Margaret Atwood", "release_date": "1985-06-01", "page_count": 311}
'
$curl_with_prefix  -X GET "https://localhost:9200/books/_search?pretty" -H 'Content-Type: application/json' -d'
{
  "query": {
    "match": {
      "name": "brave hahaha"
    }
  }
}
'

# $curl_with_prefix -X GET "https://localhost:9200/books/_search?pretty"