from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import desc, count, sum, rank, row_number, lower


def get_table(spark: SparkSession, db_table: str):
    df = spark.read \
        .format('jdbc') \
        .option('url', 'jdbc:postgresql://localhost:5432/pagila') \
        .option('driver', 'org.postgresql.Driver') \
        .option('dbtable', db_table) \
        .option('user', 'postgres') \
        .option('password', 'root')
    return df.load()


def get_count_films_in_each_category(spark: SparkSession):
    film_category_df = get_table(spark, 'film_category')
    category_df = get_table(spark, 'category')

    df = film_category_df.join(category_df, film_category_df['category_id'] == category_df['category_id'], 'inner')
    df = df.select(df['name'].alias('category_name')).groupBy('category_name').count().sort(desc('count'))
    return df.select(df['category_name'], df['count'].alias('film_count'))


def get_top_10_actors(spark: SparkSession):
    rental_df = get_table(spark, 'rental')
    inventory_df = get_table(spark, 'inventory')
    film_actor_df = get_table(spark, 'film_actor')
    actor_df = get_table(spark, 'actor')

    df = rental_df \
        .join(inventory_df, rental_df['inventory_id'] == inventory_df['inventory_id'], 'inner') \
        .join(film_actor_df, inventory_df['film_id'] == film_actor_df['film_id'], 'inner')
    df = df.groupBy(df['actor_id']).agg(count(df['actor_id']).alias('rental_count'))
    df = df.join(actor_df, df['actor_id'] == actor_df['actor_id'], 'inner').sort(desc('rental_count'))
    return df.select(df['first_name'], df['last_name']).limit(10)


def get_the_most_spent_category(spark: SparkSession):
    payment_df = get_table(spark, 'payment')
    rental_df = get_table(spark, 'rental')
    inventory_df = get_table(spark, 'inventory')
    film_category_df = get_table(spark, 'film_category')
    category_df = get_table(spark, 'category')

    df = payment_df \
        .join(rental_df, payment_df['rental_id'] == rental_df['rental_id'], 'inner') \
        .join(inventory_df, rental_df['inventory_id'] == inventory_df['inventory_id']) \
        .join(film_category_df, inventory_df['film_id'] == film_category_df['film_id'])
    df = df.groupBy(df['category_id']).agg(sum(df['amount']).alias('amount'))
    df = df.join(category_df, df['category_id'] == category_df['category_id']).sort(desc('amount'))
    return df.select(df['name'].alias('category_name')).limit(1)


def get_films_that_are_not_in_inventory(spark: SparkSession):
    film_df = get_table(spark, 'film')
    inventory_df = get_table(spark, 'inventory')

    df = film_df.join(inventory_df, film_df['film_id'] == inventory_df['film_id'], 'left')
    return df.where(inventory_df['film_id'].isNull()).select(df['title'])


def get_top_3_actors_in_children_category(spark: SparkSession):
    film_actor_df = get_table(spark, 'film_actor')
    film_category_df = get_table(spark, 'film_category')
    category_df = get_table(spark, 'category')
    actor_df = get_table(spark, 'actor')

    df = film_actor_df \
        .join(film_category_df, film_actor_df['film_id'] == film_category_df['film_id'], 'inner') \
        .join(category_df, film_category_df['category_id'] == category_df['category_id'], 'inner')
    df = df.where(df['name'] == 'Children').groupBy(df['actor_id'], df['name']).count() \
        .withColumn('num', rank().over(Window.partitionBy(df['name']).orderBy(desc('count'))))
    df = df.where(df['num'] <= 3).join(actor_df, df['actor_id'] == actor_df['actor_id'], 'inner')
    return df.select(df['first_name'], df['last_name'])


def get_citys_with_active_inactive_clients(spark: SparkSession):
    address_df = get_table(spark, 'address')
    customer_df = get_table(spark, 'customer')
    city_df = get_table(spark, 'city')

    df = address_df.join(customer_df, address_df['address_id'] == customer_df['address_id'], 'inner')
    active_df = df.where(df['active'] == 1).groupBy(df['city_id']).agg(count(df['customer_id']).alias('active_count'))
    inactive_df = df.where(df['active'] == 0).groupBy(df['city_id']).agg(count(df['customer_id']).alias('inactive_count'))
    df = city_df \
        .join(active_df, city_df['city_id'] == active_df['city_id'], 'left') \
        .join(inactive_df, city_df['city_id'] == inactive_df['city_id'], 'left')
    df = df.na.fill(value=0, subset=['active_count', 'inactive_count'])
    return df.select(df['city'], df['active_count'], df['inactive_count']).sort(desc(df['inactive_count']))


def get_category_with_the_largest_rental_amount(spark: SparkSession):
    customer_df = get_table(spark, 'customer')
    address_df = get_table(spark, 'address')
    inventory_df = get_table(spark, 'inventory')
    film_df = get_table(spark, 'film')
    film_category_df = get_table(spark, 'film_category')
    city_df = get_table(spark, 'city')
    category_df = get_table(spark, 'category')

    df = customer_df \
        .join(address_df, customer_df['address_id'] == address_df['address_id'], 'inner') \
        .join(inventory_df, customer_df['store_id'] == inventory_df['store_id'], 'inner') \
        .join(film_df, inventory_df['film_id'] == film_df['film_id'], 'inner') \
        .join(film_category_df, inventory_df['film_id'] == film_category_df['film_id'])
    df = df.groupBy(df['city_id'], df['category_id']).agg(sum(df['rental_duration']).alias('sum_rental'))
    df = df.join(city_df, df['city_id'] == city_df['city_id'], 'inner') \
        .where((lower(city_df['city']).like('a%')) | (city_df['city'].like('%-%'))) \
        .withColumn('num', row_number().over(Window.partitionBy(df['city_id']).orderBy(desc('sum_rental'))))
    df = df.join(category_df, df['category_id'] == category_df['category_id'], 'inner').where(df['num'] == 1)
    return df.select(df['city'], df['name'].alias('category_name'))


def main():
    spark = SparkSession.builder.config('spark.jars', 'postgresql-42.4.1.jar') \
        .master('local').appName('spark_framework_task').getOrCreate()

    df = get_count_films_in_each_category(spark)
    df.show(truncate=False)

    df = get_top_10_actors(spark)
    df.show(truncate=False)

    df = get_the_most_spent_category(spark)
    df.show(truncate=False)

    df = get_films_that_are_not_in_inventory(spark)
    df.show(truncate=False)

    df = get_top_3_actors_in_children_category(spark)
    df.show(truncate=False)

    df = get_citys_with_active_inactive_clients(spark)
    df.show(truncate=False)

    df = get_category_with_the_largest_rental_amount(spark)
    df.show(truncate=False)


if __name__ == '__main__':
    main()
