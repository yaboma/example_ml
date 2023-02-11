import sqlalchemy

metadata = sqlalchemy.MetaData()


mas_table = sqlalchemy.Table(
    "mas_predicted",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column(
        "nrot", sqlalchemy.String(9), unique=False, index=True, nullable=False
    ),
    sqlalchemy.Column("dt", sqlalchemy.DateTime, unique=False, index=True),
    sqlalchemy.Column(
        "num_experiments", sqlalchemy.Integer, unique=False, index=True
    ),
    sqlalchemy.Column("mas_val_up", sqlalchemy.Integer),
    sqlalchemy.Column(
        "mas_ang_up",
        sqlalchemy.Integer,
        sqlalchemy.CheckConstraint("mas_ang_up>=0 and mas_ang_up<=360"),
    ),
    sqlalchemy.Column("mas_length_up", sqlalchemy.Integer),
    sqlalchemy.Column("mas_height_up", sqlalchemy.Integer),
    sqlalchemy.Column("mas_val_down", sqlalchemy.Integer),
    sqlalchemy.Column(
        "mas_ang_down",
        sqlalchemy.Integer,
        sqlalchemy.CheckConstraint("mas_ang_down>=0 and mas_ang_down<=360"),
    ),
    sqlalchemy.Column("mas_length_down", sqlalchemy.Integer),
    sqlalchemy.Column("mas_height_down", sqlalchemy.Integer),
    sqlalchemy.Column("mas_val_middle", sqlalchemy.Integer),
    sqlalchemy.Column(
        "mas_ang_middle",
        sqlalchemy.Integer,
        sqlalchemy.CheckConstraint(
            "mas_ang_middle>=0 and mas_ang_middle<=360"
        ),
    ),
    sqlalchemy.Column("mas_length_middle", sqlalchemy.Integer),
    sqlalchemy.Column("mas_height_middle", sqlalchemy.Integer),
    sqlalchemy.Column("mas_val_middle_up", sqlalchemy.Integer),
    sqlalchemy.Column(
        "mas_ang_middle_up",
        sqlalchemy.Integer,
        sqlalchemy.CheckConstraint(
            "mas_ang_middle_up>=0 and mas_ang_middle_up<=360"
        ),
    ),
    sqlalchemy.Column("mas_length_middle_up", sqlalchemy.Integer),
    sqlalchemy.Column("mas_height_middle_up", sqlalchemy.Integer),
    sqlalchemy.Column("mas_val_middle_down", sqlalchemy.Integer),
    sqlalchemy.Column(
        "mas_ang_middle_down",
        sqlalchemy.Integer,
        sqlalchemy.CheckConstraint(
            "mas_ang_middle_down>=0 and mas_ang_middle_down<=360"
        ),
    ),
    sqlalchemy.Column("mas_length_middle_down", sqlalchemy.Integer),
    sqlalchemy.Column("mas_height_middle_down", sqlalchemy.Integer),
)
