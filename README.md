# Missing Data Interpolation

This tool fills gaps in daily weather records using inverse-distance weighted (IDW) interpolation. For each station (BARB, SCHI, MOND, TORR) and each variable (mean, max, and min temperature plus precipitation), it computes weighted averages from neighboring stations based on geographic distance, producing a continuous, quality-controlled dataset ready for downstream climate or agricultural analysis. The tool is invoked in the form of a Task within a workflow Process via the respective API call. 

## Tool Invocation Example

An example spec for executing an autonomous instance of MDI (Missing Data Interpolation) through the API would be:

```json
{
    "process_id": "f9645b89-34e4-4de2-8ecd-dc10163d9aed",
    "name": "Missing Data Interpolation",
    "image": "petroud/mdi:latest",
    "inputs": {
        "meteo_file": [
            "413f30a3-4653-4b70-8da9-99d7659b23c0"
        ],
        "coords_file":[
            "87668a7e-8d88-4074-a7f2-1a502f40c659"
        ],
        "ground_truth":[
            "1e746272-b000-4fc8-9a83-c414f6651082"
        ]
    },
    "datasets": {
        "d0": "0fc717e0-5567-4943-a843-9be47aed6eb9"
    },
    "parameters": {},
    "outputs": {
        "interpolated_file": {
            "url": "s3://abaco-bucket/MISSING_DATA/interpolated.xlsx",
            "dataset": "d0",
            "resource": {
                "name": "Interpolated Meteo Station Data",
                "relation": "owned"
            }
        }
    }
}
```

## Tool Input JSON
At runtime the tool expects the following, translated by the API, JSON: 
```json

{
        "input": {
            "coords_file": [
                "s3://abaco-bucket/MISSING_DATA/COORD.xlsx"
            ],
            "meteo_file": [
                "s3://abaco-bucket/MISSING_DATA/DATI_NETSENS_TORREVILLA.xlsx"
            ],
            "ground_truth":[
                "s3://abaco-bucket/MISSING_DATA/DATI_NETSENS_TORREVILLA_COMPLET.xlsx"
            ]
        },
        "minio": {
            "endpoint_url": "https://minio.stelar.gr",
            "id": "XXXXXXXXXX",
            "key": "XXXXXXXXXX",
            "skey": "XXXXXXXXXX"
        },
        "output": {
            "interpolated_file": "s3://abaco-bucket/MISSING_DATA/interpolated.xlsx"
        },
        "parameters": {}
}
```
### `input`
The tool expect two inputs during runtime that are being utilized in conjuction during the calculation:
- `coords_file` (XLSX): An Excel sheet holding daily observations for each station (BARB, SCHI, MOND, TORR) across four variables—mean, max, min temperature and rainfall—where some cells are blank.
- `meteo_file` (XLSX): A small Excel table that maps every station code to its latitude and longitude, supplying the distances the IDW algorithm needs to weight neighbouring stations.


### `output`
- `interpolated_file`: A new Excel file identical in structure to the raw input but with every previously missing value replaced by an inverse-distance-weighted estimate, giving a complete, gap-free time series ready for downstream analysis.


## Tool Output JSON

```json
{
    "message": "Tool executed successfully!",
    "output": {
        "interpolated_file": "s3://abaco-bucket/MISSING_DATA/interpolated.xlsx"
    },
    "metrics": {},
    "status": "success"
}
```


## How to build 
Alter the `IMGTAG` in Makefile with a repository from your Image Registry and hit 
`make` in your terminal within the same directory.
