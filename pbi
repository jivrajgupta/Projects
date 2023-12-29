ColorCode = 
    SWITCH(
        TRUE(),
        'YourTableName'[Column1] = "Condition1" && 'YourTableName'[Column2] = "ConditionA", "#FF0000",  -- Red
        'YourTableName'[Column1] = "Condition2" && 'YourTableName'[Column2] = "ConditionB", "#00FF00",  -- Green
        'YourTableName'[Column1] = "Condition3" && 'YourTableName'[Column2] = "ConditionC", "#0000FF",  -- Blue
        'YourTableName'[Column1] = "Condition4" && 'YourTableName'[Column2] = "ConditionD", "#FFFF00",  -- Yellow
        'YourTableName'[Column1] = "Condition5" && 'YourTableName'[Column2] = "ConditionE", "#FF00FF",  -- Magenta
        "#FFFFFF"  -- Default color (white) for other cases
    )
