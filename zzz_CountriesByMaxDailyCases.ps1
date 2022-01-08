function CountriesByMaxDailyCases()
{
        $csv=Import-Csv ".\covid-19-data\public\data\owid-covid-data.csv"
        foreach($record in $csv)
        {
                $record.new_cases=[int]$record.new_cases
        }
        
        $unique_countries = $csv | Select-Object -Unique location
        
        foreach($country in $unique_countries)
        {
                
        }
        
}

CountriesByMaxDailyCases